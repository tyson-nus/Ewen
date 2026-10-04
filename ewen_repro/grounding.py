from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
import numpy as np
from scipy import signal
from .metrics import BANDS, canonical_sha256, parse_withheld
ATTRIBUTES = ('temporal_delta', 'temporal_theta', 'temporal_alpha', 'temporal_beta', 'frontal_posterior_alpha', 'o1_o2_alpha', 'c3_c4_alpha_beta')
FRONTAL = ('FP1', 'FP2', 'F3', 'F4', 'FZ')
POSTERIOR = ('P3', 'P4', 'PZ', 'O1', 'O2')
BAND_LIMITS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))
CLAUSES = {**{f'temporal_{band}': {'negative': f'{band} relative power decreases from the first half to the second half.', 'neutral': f'{band} relative power is similar between the first half and the second half.', 'positive': f'{band} relative power increases from the first half to the second half.'} for band in BANDS[:4]}, 'frontal_posterior_alpha': {'negative': 'Posterior alpha relative power is lower than frontal alpha relative power.', 'neutral': 'Posterior and frontal alpha relative power are similar.', 'positive': 'Posterior alpha relative power is higher than frontal alpha relative power.'}, 'o1_o2_alpha': {'negative': 'O2 alpha relative power is lower than O1 alpha relative power.', 'neutral': 'O1 and O2 alpha relative power are similar.', 'positive': 'O2 alpha relative power is higher than O1 alpha relative power.'}, 'c3_c4_alpha_beta': {'negative': 'C4 combined alpha-beta relative power is lower than C3 combined alpha-beta relative power.', 'neutral': 'C3 and C4 combined alpha-beta relative power are similar.', 'positive': 'C4 combined alpha-beta relative power is higher than C3 combined alpha-beta relative power.'}}

@dataclass(frozen=True)
class GroundingConfig:
    sample_rate_hz: int = 200
    patch_samples: int = 200
    state_threshold: float = 0.05
    welch_seconds: float = 2.0
    welch_overlap_fraction: float = 0.5
    psd_window: str = 'hann'
    psd_detrend: str = 'constant'

    def __post_init__(self):
        if self.sample_rate_hz != 200 or self.patch_samples != 200 or self.state_threshold != 0.05:
            raise ValueError('Paper grounding requires 200Hz, 200-sample patches and fixed 0.05 state boundaries')
        if self.welch_seconds <= 0 or not 0 <= self.welch_overlap_fraction < 1:
            raise ValueError('Invalid explicit Welch reconstruction settings')

    @property
    def fingerprint(self):
        return canonical_sha256({'config': asdict(self), 'attributes': ATTRIBUTES, 'clauses': CLAUSES, 'normalization_band': [0.5, 45], 'frontal': FRONTAL, 'posterior': POSTERIOR, 'contrast_order': 'first_to_second; frontal_to_posterior; O1_to_O2; C3_to_C4'})

def _array(value: Any, dtype=None):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)

def _canonical_channel(name: str) -> str:
    name = re.sub('^EEG\\s+', '', str(name).strip(), flags=re.I).upper()
    name = re.sub('-(REF|LE|AVG)$', '', name)
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    return '-'.join((aliases.get(part, part) for part in name.split('-')))

def channel_name_map(dataset, row: Mapping[str, Any]) -> dict[int, str]:
    if getattr(dataset, 'channel_identity', 'stored') == 'canonical':
        vocab = getattr(dataset, 'canonical_vocab', None)
        if not vocab:
            raise ValueError('Canonical channel identity requires a corpus canonical_vocab')
        return {int(v): _canonical_channel(k) for (k, v) in vocab.items() if k != '[PAD]'}
    vocab = getattr(dataset, 'channel_vocab', None)
    if not vocab:
        raise ValueError('Stored channel identity requires the corpus channel_vocab')
    names = row.get('channel_names')
    if not isinstance(names, list) or not names:
        raise ValueError('Grounding requires independent channel-name metadata')
    if any((name not in vocab for name in names)):
        raise ValueError('Record channel names absent from corpus vocabulary')
    return {int(vocab[name]): _canonical_channel(name) for name in names}

def reconstruct_waveform(sample: Mapping[str, Any], id_to_name: Mapping[int, str], config: GroundingConfig | None=None) -> tuple[np.ndarray, list[str], np.ndarray]:
    cfg = config or GroundingConfig()
    patches = _array(sample['eeg'], np.float64)
    mask = _array(sample['input_mask'])
    channels = _array(sample['input_chans'])
    times = _array(sample['input_times'])
    n = len(patches)
    if patches.ndim != 2 or patches.shape[1] != cfg.patch_samples or mask.shape != (n,) or (mask.dtype.kind != 'b') or (channels.shape != (n,)) or (channels.dtype.kind not in 'iu') or (times.shape != (n,)) or (times.dtype.kind not in 'iu') or (not n):
        raise ValueError('EEG patch coordinates must be [M,200], boolean mask and integer channel/time arrays')
    if np.any(times < 0) or np.any(channels <= 0) or int(times.max()) >= 64:
        raise ValueError('Invalid temporal/channel patch coordinates')
    n_times = int(times.max()) + 1
    ids = sorted(set((int(c) for c in channels)))
    if any((c not in id_to_name for c in ids)):
        raise ValueError('Patch channel ID is absent from independent channel metadata')
    x = np.zeros((len(ids), n_times * cfg.patch_samples), dtype=np.float64)
    validity = np.zeros(len(ids), dtype=bool)
    for (channel_index, cid) in enumerate(ids):
        indices = np.flatnonzero(channels == cid)
        if len(set((int(t) for t in times[indices]))) != len(indices):
            raise ValueError('Duplicate channel/time patch pair')
        complete = set((int(t) for t in times[indices])) == set(range(n_times))
        valid = complete and all(mask[indices]) and np.all(np.isfinite(patches[indices]))
        for token in indices:
            start = int(times[token]) * cfg.patch_samples
            if mask[token] and np.all(np.isfinite(patches[token])):
                x[channel_index, start:start + cfg.patch_samples] = patches[token]
        validity[channel_index] = valid and np.std(x[channel_index]) > 1e-12
    return (x, [_canonical_channel(id_to_name[c]) for c in ids], validity)

def _integral(f: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    interior = (f > low) & (f < high)
    frequencies = np.r_[low, f[interior], high]
    values = np.r_[np.interp(low, f, power), power[interior], np.interp(high, f, power)]
    return float(np.sum(np.diff(frequencies) * (values[:-1] + values[1:]) / 2))

def _relative_power(x: np.ndarray, band: tuple[float, float], cfg: GroundingConfig) -> float | None:
    if x.ndim != 2 or not len(x) or x.shape[1] < 8 or (not np.all(np.isfinite(x))):
        return None
    if np.any(np.std(x, axis=1) <= 1e-12):
        return None
    segment = min(x.shape[1], max(8, int(round(cfg.welch_seconds * cfg.sample_rate_hz))))
    (frequencies, powers) = signal.welch(x, fs=cfg.sample_rate_hz, window=cfg.psd_window, nperseg=segment, noverlap=int(segment * cfg.welch_overlap_fraction), detrend=cfg.psd_detrend, axis=-1)
    power = powers.mean(axis=0)
    if not np.all(np.isfinite(power)):
        return None
    denominator = _integral(frequencies, power, 0.5, 45)
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    value = _integral(frequencies, power, *band) / denominator
    return value if math.isfinite(value) else None

def contrast_state(u: float | None, v: float | None, threshold: float=0.05) -> tuple[str | None, float | None]:
    if threshold != 0.05:
        raise ValueError('Paper comparison boundaries are fixed at 0.05')
    if u is None or v is None or (not np.isfinite(u)) or (not np.isfinite(v)) or (u + v <= 0):
        return (None, None)
    value = float((v - u) / (u + v))
    tol = 8 * np.finfo(np.float64).eps
    state = 'negative' if value < -threshold - tol else 'positive' if value > threshold + tol else 'neutral'
    return (state, value)

def waveform_targets(eeg: np.ndarray, channel_names: list[str], channel_validity: np.ndarray, config: GroundingConfig | None=None) -> dict:
    cfg = config or GroundingConfig()
    x = np.asarray(eeg, dtype=np.float64)
    valid = np.asarray(channel_validity, dtype=bool)
    names = [_canonical_channel(name) for name in channel_names]
    if x.ndim != 2 or len(names) != len(x) or valid.shape != (len(x),):
        raise ValueError('Waveform, channel names and validity dimensions disagree')
    available = {a: False for a in ATTRIBUTES}
    (values, measurements) = ({}, {})
    half = x.shape[1] // 2
    if half >= 8 and x.shape[1] % 2 == 0:
        same = valid & np.all(np.isfinite(x), axis=1)
        same &= (np.std(x[:, :half], axis=1) > 1e-12) & (np.std(x[:, half:], axis=1) > 1e-12)
        for (band, limits) in zip(BANDS[:4], BAND_LIMITS[:4]):
            a = f'temporal_{band}'
            u = _relative_power(x[same, :half], limits, cfg)
            v = _relative_power(x[same, half:], limits, cfg)
            (state, contrast) = contrast_state(u, v)
            if state is not None:
                (available[a], values[a]) = (True, state)
                measurements[a] = {'first': u, 'second': v, 'contrast': contrast, 'valid_channel_count': int(same.sum())}
    unique_names = {name: i for (i, name) in enumerate(names) if names.count(name) == 1 and valid[i]}
    alpha = {name: _relative_power(x[i:i + 1], (8, 13), cfg) for (name, i) in unique_names.items()}
    frontal = [alpha[name] for name in FRONTAL if alpha.get(name) is not None]
    posterior = [alpha[name] for name in POSTERIOR if alpha.get(name) is not None]
    if len(frontal) >= 2 and len(posterior) >= 2:
        (u, v) = (float(np.mean(frontal)), float(np.mean(posterior)))
        (state, contrast) = contrast_state(u, v)
        if state is not None:
            a = 'frontal_posterior_alpha'
            (available[a], values[a]) = (True, state)
            measurements[a] = {'first': u, 'second': v, 'contrast': contrast, 'frontal_channel_count': len(frontal), 'posterior_channel_count': len(posterior)}
    for (a, first, second, band) in (('o1_o2_alpha', 'O1', 'O2', (8, 13)), ('c3_c4_alpha_beta', 'C3', 'C4', (8, 30))):
        if first not in unique_names or second not in unique_names:
            continue
        u = _relative_power(x[unique_names[first]:unique_names[first] + 1], band, cfg)
        v = _relative_power(x[unique_names[second]:unique_names[second] + 1], band, cfg)
        (state, contrast) = contrast_state(u, v)
        if state is not None:
            (available[a], values[a]) = (True, state)
            measurements[a] = {'first': u, 'second': v, 'contrast': contrast}
    return {'values': values, 'available': available, 'measurements': measurements}

def clause_bank() -> dict:
    return {a: {state: [re.escape(sentence.casefold())] for (state, sentence) in states.items()} for (a, states) in CLAUSES.items()}

def parse_summary_states(text: str) -> dict:
    return parse_withheld(text, clause_bank())

def _raw_descriptor_reference(sample: Mapping[str, Any]) -> tuple[dict, dict]:
    raw = _array(sample['raw_descriptors'], np.float64)
    mask = _array(sample['raw_descriptor_mask'])
    if raw.shape != (22,) or mask.shape != (22,) or mask.dtype.kind != 'b':
        raise ValueError('Direct references require unstandardized 22 descriptors and their observed-value masks')
    available = {band: bool(mask[i] and np.isfinite(raw[i]) and (0 <= raw[i] <= 1)) for (i, band) in enumerate(BANDS)}
    values = {band: float(raw[i]) for (i, band) in enumerate(BANDS) if available[band]}
    return (values, available)

def _relation_text(sample: Mapping[str, Any]) -> tuple[str, str]:
    raw = _array(sample['raw_descriptors'], np.float64)
    observed = _array(sample['raw_descriptor_mask'], bool)
    modalities = _array(sample['modality_mask'])
    if modalities.shape != (3,) or modalities.dtype.kind != 'b':
        raise ValueError('Auxiliary relation target requires actual boolean modality availability')
    names = ('EOG', 'ECG', 'EMG')
    present = [name for (name, flag) in zip(names, modalities) if flag]
    if not present:
        return ('Auxiliary physiological relation assessment has insufficient evidence because EOG, ECG and EMG are unavailable.', 'insufficient evidence')
    coupling = (('EOG-EEG maximum lagged correlation', 10), ('EOG-EEG regression gain', 11), ('ECG-EEG HEP amplitude', 15), ('ECG-EEG PLV', 16), ('EMG-EEG beta coherence', 20), ('EMG-EEG phase-slope delay', 21))
    evidence = [f'{name} is {raw[i]:.6g}' for (name, i) in coupling if observed[i] and np.isfinite(raw[i])]
    sentence = 'Available auxiliary modalities are ' + ', '.join(present) + '.'
    if evidence:
        sentence += ' Observed coupling descriptors: ' + '; '.join(evidence) + '.'
    else:
        sentence += ' Available auxiliary coupling descriptors provide insufficient evidence for a categorical relation state.'
    return (sentence, 'numerical evidence without unspecified categorical bins' if evidence else 'insufficient evidence')

def _text_targets(sample: Mapping[str, Any], direct: dict, withheld: dict) -> tuple[dict, dict]:
    label = str(sample.get('label_target', '')).strip()
    if not label:
        raise ValueError('Supervised task-label target requires dataset label metadata')
    available_bands = [band for band in BANDS if direct['available'][band]]
    dominant = max(available_bands, key=lambda b: direct['values'][b]) if available_bands else None
    band_text = f'The dominant EEG band is {dominant}.' if dominant else 'Dominant EEG band assessment has insufficient evidence.'
    (relation_text, relation_state) = _relation_text(sample)
    numeric = ' '.join((f"{band.capitalize()} relative power is {direct['values'][band]:.6f}." for band in available_bands))
    comparisons = ' '.join((CLAUSES[a][withheld['values'][a]] for a in ATTRIBUTES if withheld['available'][a]))
    if not comparisons:
        comparisons = 'The required waveform comparisons are unavailable for this window.'
    summary = f'The dataset annotation is {label}. {band_text} {numeric} {relation_text} {comparisons}'
    texts = {'label': label, 'band': band_text, 'relation': relation_text, 'summary': summary}
    fields = {'label': label, 'band': dominant, 'relation': relation_state, 'summary': {'label': label, 'band': dominant, 'relation': relation_state}}
    return (texts, fields)

def _sample_hash(dataset_name: str, sample_key: str) -> str:
    return hashlib.sha256(f'{dataset_name}/{sample_key}'.encode()).hexdigest()

def _write_json(path: Path, value: Any):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

def build_grounding(train, val, test, output, *, config: GroundingConfig | None=None) -> dict:
    cfg = config or GroundingConfig()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    datasets = (('train', train), ('val', val), ('test', test))
    names = set()
    for (split, dataset) in datasets:
        if getattr(dataset, 'split', None) != split:
            raise ValueError('Grounding datasets must be supplied in train/val/test order')
        if not hasattr(dataset, 'records') or len(dataset.records) != len(dataset):
            raise ValueError('Grounding requires an independent record manifest for every waveform')
        names.update((str(r['dataset_name']) for r in dataset.records))
    if any((not re.fullmatch('[A-Za-z0-9_ -]+', name) for name in names)):
        raise ValueError('Dataset names must be sanitized identifiers')
    (handles, paths) = ({}, {})
    direct_rows = {name: [] for name in names}
    withheld_rows = {name: [] for name in names}
    seen = set()
    diagnostics = {name: {'counts': Counter(), 'availability': Counter(), 'unavailable': Counter(), 'missing_direct': Counter(), 'discarded_tail_windows': 0} for name in names}
    try:
        for name in names:
            folder = destination / name
            folder.mkdir(parents=True, exist_ok=True)
            paths[name] = folder / 'independent_targets.jsonl'
            handles[name] = paths[name].with_suffix('.jsonl.tmp').open('w')
        for (split, dataset) in datasets:
            for (i, row) in enumerate(dataset.records):
                sample = dataset[i]
                (name, key) = (str(row['dataset_name']), str(row['sample_key']))
                sid = _sample_hash(name, key)
                if sid in seen:
                    raise ValueError('Same EEG window occurs in multiple grounding splits')
                seen.add(sid)
                if sample.get('sample_id') != sid or sample.get('split') != split:
                    raise ValueError('Grounding sample identity/split disagrees with independent manifest')
                (x, channels, valid) = reconstruct_waveform(sample, channel_name_map(dataset, row), cfg)
                withheld = waveform_targets(x, channels, valid, cfg)
                (values, available) = _raw_descriptor_reference(sample)
                direct = {'values': values, 'available': available}
                (texts, fields) = _text_targets(sample, direct, withheld)
                record = {'sample_key': key, 'sample_id': sid, 'dataset': name, 'split': split, 'targets': texts, 'fields': fields, 'provenance': 'independent_waveform_measurements', 'target_construction': 'deterministic templates; no fluent-LLM refinement', 'label_provenance': 'supervised dataset annotation; excluded from model features', 'relation_encoding': 'Explicit reconstruction choice; manuscript categorical relation bins absent', 'grounding_config_sha256': cfg.fingerprint}
                handles[name].write(json.dumps(record, allow_nan=False) + '\n')
                diagnostics[name]['counts'][split] += 1
                diagnostics[name]['discarded_tail_windows'] += bool(sample.get('discarded_tail_samples', 0))
                if split == 'test':
                    direct_rows[name].append({'sample_id': sid, **direct})
                    withheld_rows[name].append({'sample_id': sid, **withheld})
                    diagnostics[name]['availability'].update((a for (a, flag) in withheld['available'].items() if flag))
                    diagnostics[name]['unavailable'].update((a for (a, flag) in withheld['available'].items() if not flag))
                    diagnostics[name]['missing_direct'].update((b for (b, flag) in available.items() if not flag))
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in paths.values():
            path.with_suffix('.jsonl.tmp').unlink(missing_ok=True)
        raise
    for (name, handle) in handles.items():
        handle.close()
        paths[name].with_suffix('.jsonl.tmp').replace(paths[name])
    report = {'schema': 1, 'provenance': 'independent_waveform_measurements', 'grounding_config': asdict(cfg), 'grounding_config_sha256': cfg.fingerprint, 'state_definition': '(second-first)/(first+second); fixed +/-0.05; equality neutral', 'normalization_band_hz': [0.5, 45], 'parser_coverage': 'Frozen exact template clauses; semantic aliases absent from manuscript', 'reconstruction_choices': ['Welch Hann window, two-second segments, 50% overlap, constant detrending, interpolated trapezoid integration', 'Regional means of per-channel relative alpha power', 'Deterministic prose templates without off-the-shelf LLM refinement', 'Relations report observed numerical evidence; no unspecified low/high coupling bins'], 'human_expert_evaluation': 'Absent; machine waveform targets are not expert ratings', 'conditions': {}}
    for name in sorted(names):
        folder = destination / name
        shared = {'frozen': True, 'independent': True, 'split': 'test', 'dataset': name, 'grounding_config_sha256': cfg.fingerprint, 'provenance': 'independent_waveform_measurements'}
        direct = {**shared, 'kind': 'direct_numeric', 'attributes': list(BANDS), 'records': direct_rows[name]}
        withheld = {**shared, 'kind': 'withheld_states', 'attributes': list(ATTRIBUTES), 'records': withheld_rows[name], 'parser_clause_bank': clause_bank(), 'attribute_ordered_measurements': {**{a: ['first half', 'second half'] for a in ATTRIBUTES[:4]}, 'frontal_posterior_alpha': ['frontal', 'posterior'], 'o1_o2_alpha': ['O1', 'O2'], 'c3_c4_alpha_beta': ['C3', 'C4']}}
        _write_json(folder / 'direct_reference.json', direct)
        _write_json(folder / 'withheld_reference.json', withheld)
        detail = diagnostics[name]
        report['conditions'][name] = {'targets_path': f'{name}/independent_targets.jsonl', 'targets_sha256': hashlib.sha256(paths[name].read_bytes()).hexdigest(), 'direct_numeric': {'path': f'{name}/direct_reference.json', 'sha256': canonical_sha256(direct)}, 'withheld_states': {'path': f'{name}/withheld_reference.json', 'sha256': canonical_sha256(withheld)}, 'counts': dict(detail['counts']), 'test_available_targets': dict(detail['availability']), 'test_unavailable_targets': dict(detail['unavailable']), 'test_missing_direct_targets': dict(detail['missing_direct']), 'discarded_tail_windows': detail['discarded_tail_windows']}
    _write_json(destination / 'grounding_report.json', report)
    return report
