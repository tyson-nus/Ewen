from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import warnings
import h5py
import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset as TorchDataset
from .descriptors import DescriptorConfig, DescriptorStandardizer, descriptor_text, descriptor_vector
DATASET_NAMES = ('TUAB', 'TUEV', 'mental-arithmetic', 'BCICIV_2a', 'SHU-MI', 'Speech', 'mumtaz', 'SEEDV')
DEFAULT_DATASET_NAMES = DATASET_NAMES
TARGET_SFREQ = 200
PATCH_SIZE = 200
PAPER = {'TUAB': dict(seconds=10, channels=23, classes=2, subjects=2329, windows=409455), 'TUEV': dict(seconds=5, channels=23, classes=6, subjects=370, windows=113353), 'mental-arithmetic': dict(seconds=5, channels=19, classes=2, subjects=36, windows=1707), 'SHU-MI': dict(seconds=4, channels=32, classes=2, subjects=25, windows=11988), 'mumtaz': dict(seconds=5, channels=19, classes=2, subjects=64, windows=7083), 'BCICIV_2a': dict(seconds=4, channels=22, classes=4, subjects=9, windows=5088), 'Speech': dict(seconds=3, channels=64, classes=5, subjects=15, windows=6000), 'SEEDV': dict(seconds=1, channels=62, classes=5, subjects=15, windows=117744)}
QUESTIONS = {'TUAB': 'Classify this EEG window as normal or abnormal.', 'TUEV': 'Classify the EEG event as SPSW, GPED, PLED, EYEM, ARTF, or BCKG.', 'mental-arithmetic': 'Classify this EEG window as rest or mental arithmetic.', 'SHU-MI': 'Classify the motor imagery as left hand or right hand.', 'mumtaz': 'Classify the recording as healthy control or major depressive disorder.', 'BCICIV_2a': 'Classify the motor imagery as left hand, right hand, foot, or tongue.', 'Speech': 'Classify the imagined phrase as hello, help me, stop, thank you, or yes.', 'SEEDV': 'Classify the emotion as disgust, fear, sadness, neutral, or happiness.'}
_PUBLIC_KEYS = ('dataset_name', 'sample_key', 'session_key', 'subject_id', 'recording_id', 'window_index', 'label_id', 'label_text', 'orig_sfreq_hz', 'stored_sfreq_hz', 'window_duration_sec', 'signal_shape', 'channel_names', 'split')

def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def normalize_split(split):
    aliases = {'valid': 'val', 'validation': 'val', 'eval': 'val', 'training': 'train', 'testing': 'test'}
    result = aliases.get(str(split).lower(), str(split).lower())
    if result not in ('train', 'val', 'test'):
        raise ValueError(f'Unsupported split {result!r}')
    return result

def normalized_subject(name, subject):
    result = re.sub('\\s+', '-', str(subject).strip())
    if name == 'Speech':
        result = re.sub('^(train|val|valid|validation|test)[-_]', '', result, flags=re.I)
    return result.casefold()

def canonical_channel(name):
    name = re.sub('^EEG\\s+', '', str(name).strip(), flags=re.I).upper()
    name = re.sub('-(REF|LE|AVG)$', '', name)
    legacy = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    return '-'.join((legacy.get(x, x) for x in name.split('-')))

def iter_index(path):
    with Path(path).open() as handle:
        for (line_number, line) in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or not all((k in raw for k in ('sample_key', 'split', 'subject_id'))):
                raise ValueError(f'Invalid sample index at line {line_number}')
            row = {key: raw[key] for key in _PUBLIC_KEYS if key in raw}
            row['split'] = normalize_split(row['split'])
            yield row

def audit_splits(rows, name):
    (subjects, sessions, samples) = (defaultdict(set) for _ in range(3))
    (counts, channels, rates, durations, labels) = (Counter() for _ in range(5))
    (errors, notes) = ([], [])
    bcic_bad = 0
    expected_subject_bad = 0
    seen = set()
    duplicate = 0
    for row in rows:
        split = normalize_split(row['split'])
        key = str(row['sample_key'])
        if (split, key) in seen:
            duplicate += 1
        seen.add((split, key))
        subjects[split].add(normalized_subject(name, row['subject_id']))
        sessions[split].add(str(row.get('session_key', row.get('recording_id', ''))))
        samples[split].add(key)
        counts[split] += 1
        channels[len(row.get('channel_names', []))] += 1
        rates[str(row.get('stored_sfreq_hz', row.get('orig_sfreq_hz')))] += 1
        durations[str(row.get('window_duration_sec'))] += 1
        labels[str(row.get('label_id'))] += 1
        if name == 'BCICIV_2a':
            match = re.match('A\\d+(T|E)', str(row.get('recording_id', '')), flags=re.I)
            if not match or (match[1].upper() == 'T') != (split in ('train', 'val')):
                bcic_bad += 1
        if name in ('mental-arithmetic', 'SHU-MI'):
            match = re.search('(\\d+)$', str(row['subject_id']))
            number = int(match[1]) if match else -1
            if name == 'mental-arithmetic':
                expected = 'train' if 0 <= number <= 27 else 'val' if 28 <= number <= 31 else 'test' if 32 <= number <= 35 else None
            else:
                expected = 'train' if 1 <= number <= 15 else 'val' if 16 <= number <= 20 else 'test' if 21 <= number <= 25 else None
            expected_subject_bad += split != expected
    if duplicate:
        errors.append(f'{duplicate} repeated sample keys within a split')
    overlaps = {}
    for (a, b) in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        pair = f'{a}/{b}'
        overlaps[pair] = {'subjects': len(subjects[a] & subjects[b]), 'sessions': len(sessions[a] & sessions[b]), 'samples': len(samples[a] & samples[b])}
        if overlaps[pair]['samples']:
            errors.append(f'{pair}: identical sample keys occur across splits')
        if name not in ('BCICIV_2a', 'Speech') and overlaps[pair]['subjects']:
            errors.append(f"{pair}: {overlaps[pair]['subjects']} subjects overlap in a subject-disjoint protocol")
        if name not in ('BCICIV_2a', 'Speech') and overlaps[pair]['sessions']:
            errors.append(f'{pair}: sessions overlap')
        if name == 'BCICIV_2a' and b == 'test' and overlaps[pair]['sessions']:
            errors.append(f'{pair}: BCIC session 1 and session 2 overlap')
    if name == 'BCICIV_2a':
        notes.append('Within-subject train/test is permitted; session T is train/val, session E is test; validation trials may share training sessions.')
        if bcic_bad:
            errors.append(f'{bcic_bad} trials violate the BCIC session T/E assignment')
    if name == 'Speech':
        notes.append('Within-subject and within-session overlap is permitted by the competition trial partition; split-prefixed subject names are normalized.')
    if expected_subject_bad:
        errors.append(f"{expected_subject_bad} samples violate the paper's explicit subject-ID partition")
    total_subjects = len(set().union(*subjects.values())) if subjects else 0
    if name == 'SEEDV':
        if '7' in set().union(*subjects.values()):
            errors.append('SEED-V original subject 7 is present although the paper excludes it')
        if any((len(subjects[s]) != 5 for s in ('train', 'val', 'test'))):
            errors.append('SEED-V requires five retained subjects in each partition')
    if name == 'mumtaz' and len(subjects['test']) != 11:
        errors.append('Mumtaz requires 11 test subjects')
    return {'dataset': name, 'counts': dict(counts), 'subject_counts': {s: len(v) for (s, v) in subjects.items()}, 'session_counts': {s: len(v) for (s, v) in sessions.items()}, 'total_subjects': total_subjects, 'cross_split_overlap': overlaps, 'channels': dict(channels), 'stored_rates_hz': dict(rates), 'declared_durations_seconds': dict(durations), 'label_counts': dict(labels), 'errors': errors, 'notes': notes}

def audit_dataset(root, name, manifest_path=None, sfreq_overrides=None):
    root = require_path(root, 'data_root')
    path = Path(manifest_path) if manifest_path else Path(root) / name / 'aligned_samples.jsonl'
    report = audit_splits(iter_index(path), name)
    meta_errors = Counter()
    metadata_warnings = Counter()
    rate_override = (sfreq_overrides or {}).get(name)
    for row in iter_index(path):
        shape = row.get('signal_shape', [])
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            meta_errors['invalid signal_shape'] += 1
            continue
        sfreq = rate_override or row.get('stored_sfreq_hz', row.get('orig_sfreq_hz'))
        duration = row.get('window_duration_sec')
        if not isinstance(sfreq, (int, float)) or not np.isfinite(sfreq) or sfreq <= 0:
            meta_errors['missing or invalid stored sampling rate'] += 1
            continue
        if rate_override is None and (not isinstance(duration, (int, float)) or abs(shape[1] / sfreq - duration) > max(1 / sfreq, 1e-05)):
            meta_errors['signal length contradicts sampling rate and declared duration; explicit audited stored-sfreq override required'] += 1
        if abs(shape[1] / sfreq - PAPER[name]['seconds']) > max(1 / sfreq, 1e-05):
            meta_errors['effective window duration differs from Appendix D'] += 1
        if shape[0] != len(row.get('channel_names', [])):
            meta_errors['channel_names do not match signal shape'] += 1
        if shape[0] != PAPER[name]['channels']:
            metadata_warnings['channel count differs from paper table'] += 1
    report['metadata_errors'] = dict(meta_errors)
    report['warnings'] = dict(metadata_warnings)
    report['errors'] += [f'{n} samples: {message}' for (message, n) in meta_errors.items()]
    report['paper_expected'] = PAPER[name]
    if report['total_subjects'] != PAPER[name]['subjects']:
        report['warnings']['subject count differs from paper table'] = report['total_subjects']
    if sum(report['counts'].values()) != PAPER[name]['windows']:
        report['warnings']['window count differs from paper table'] = sum(report['counts'].values())
    report['sfreq_override_hz'] = rate_override
    report['manifest_sha256'] = _file_hash(path)
    report['publication_ready'] = not report['errors'] and (not report['warnings'])
    return report

def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while (chunk := f.read(1 << 20)):
            digest.update(chunk)
    return digest.hexdigest()

class _JSONLIndex:

    def __init__(self, path, selected_keys):
        self.path = Path(path)
        self.offsets = {}
        self.handle = None
        self.pid = None
        if self.path.exists():
            with self.path.open('rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = row.get('sample_key')
                    if key in selected_keys:
                        if key in self.offsets:
                            raise ValueError('Duplicate sidecar records for a sample key')
                        self.offsets[key] = offset

    def get(self, key):
        if key not in self.offsets:
            return {}
        if self.handle is None or self.pid != os.getpid():
            self.close()
            self.handle = self.path.open('rb')
            self.pid = os.getpid()
        self.handle.seek(self.offsets[key])
        return json.loads(self.handle.readline())

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    def __getstate__(self):
        return {**self.__dict__, 'handle': None, 'pid': None}

def resample_eeg(eeg, source_sfreq, target_sfreq=TARGET_SFREQ):
    if not np.isfinite(source_sfreq) or not np.isfinite(target_sfreq) or source_sfreq <= 0 or (target_sfreq <= 0):
        raise ValueError('Sampling rates must be positive')
    if np.isclose(source_sfreq, target_sfreq):
        return np.asarray(eeg, dtype=np.float64)
    ratio = Fraction(float(target_sfreq) / float(source_sfreq)).limit_denominator(100000)
    return signal.resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1)

def preprocess_eeg(eeg, name, sfreq, bandpass=True, notch_hz=None):
    x = resample_eeg(np.asarray(eeg, dtype=np.float64), sfreq)
    good = np.all(np.isfinite(x), axis=1) & (np.std(x, axis=1) > 1e-12)
    x[~good] = 0
    if bandpass:
        limits = (0.5, 45.0) if name in ('mental-arithmetic', 'Speech') else (0.1, 75.0)
        sos = signal.butter(4, limits, fs=TARGET_SFREQ, btype='bandpass', output='sos')
        x = signal.sosfiltfilt(sos, x, axis=-1)
    if notch_hz is not None:
        if not 0 < notch_hz < TARGET_SFREQ / 2:
            raise ValueError('Notch frequency must fall below target Nyquist')
        (b, a) = signal.iirnotch(notch_hz, Q=30, fs=TARGET_SFREQ)
        x = signal.filtfilt(b, a, x, axis=-1)
    center = np.median(x, axis=-1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(x - center), axis=-1, keepdims=True)
    good &= scale[:, 0] > 1e-12
    normalized = np.zeros_like(x, dtype=np.float32)
    normalized[good] = ((x[good] - center[good]) / scale[good]).astype(np.float32)
    return (normalized, good)

def patchify(eeg, channel_ids, channel_mask=None):
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2 or len(channel_ids) != eeg.shape[0]:
        raise ValueError('EEG/channel identity dimensions disagree')
    seconds = eeg.shape[1] // PATCH_SIZE
    if seconds < 1:
        raise ValueError('A sample must contain at least one 200-sample patch')
    patches = eeg[:, :seconds * PATCH_SIZE].reshape(-1, PATCH_SIZE)
    chans = np.repeat(np.asarray(channel_ids, dtype=np.int64), seconds)
    times = np.tile(np.arange(seconds, dtype=np.int64), eeg.shape[0])
    valid = np.repeat(np.ones(eeg.shape[0], dtype=bool) if channel_mask is None else channel_mask, seconds)
    patches[~valid] = 0
    return (patches, valid, chans, times, eeg.shape[1] - seconds * PATCH_SIZE)

class H5EEGDataset(TorchDataset):

    def __init__(self, root, names, split, max_samples=None, sfreq_overrides=None, manifest_path=None, standardizer=None, allow_audited_mismatch=False, channel_identity='stored', bandpass=True, notch_hz=None, descriptor_config=None, targets_path=None, smoke_template_targets=False, auxiliary_policy='audited_legacy', auxiliary_signals=None, physiology_config=None):
        self.root = Path(require_path(root, 'data_root'))
        self.names = [names] if isinstance(names, str) else list(names)
        self.split = normalize_split(split)
        self.sfreq_overrides = dict(sfreq_overrides or {})
        self.standardizer = standardizer
        (self.bandpass, self.notch_hz) = (bandpass, notch_hz)
        self.descriptor_config = descriptor_config or DescriptorConfig()
        if auxiliary_policy not in ('audited_legacy', 'provided', 'raw'):
            raise ValueError('auxiliary_policy must be audited_legacy, provided, or raw')
        if auxiliary_policy == 'raw' and auxiliary_signals is None:
            raise ValueError('The raw auxiliary policy requires an aligned signal provider')
        self.auxiliary_policy = auxiliary_policy
        self.auxiliary_signals = auxiliary_signals
        if physiology_config is None:
            from .physiology import PhysiologyConfig
            physiology_config = PhysiologyConfig()
        self.physiology_config = physiology_config
        self.extraction_fingerprint = _hash(json.dumps({'eeg_descriptor_config': self.descriptor_config.fingerprint, 'auxiliary_policy': auxiliary_policy, 'physiology_config': physiology_config.fingerprint, 'bandpass': bool(bandpass), 'notch_hz': notch_hz}, sort_keys=True))
        self.smoke_template_targets = smoke_template_targets
        self.channel_identity = channel_identity
        if channel_identity not in ('stored', 'canonical'):
            raise ValueError('channel_identity must be stored or canonical')
        if standardizer and standardizer.extractor_fingerprint not in (None, self.extraction_fingerprint):
            raise ValueError('Descriptor extraction differs from training statistics')
        (self.records, self.audits) = ([], {})
        (self.h5, self.h5_pid, self.metrics, self.targets) = ({}, None, {}, {})
        vocab_path = self.root / 'channel_vocab.json'
        self.channel_vocab = json.loads(vocab_path.read_text()) if vocab_path.exists() else {}
        if not self.channel_vocab:
            all_channels = set()
            for dataset_name in self.names:
                for metadata in iter_index(self.root / dataset_name / 'aligned_samples.jsonl'):
                    all_channels.update(metadata.get('channel_names', []))
            self.channel_vocab = {'[PAD]': 0, **{channel: i + 1 for (i, channel) in enumerate(sorted(all_channels))}}
        canonical_names = sorted({canonical_channel(x) for x in self.channel_vocab if x != '[PAD]'})
        self.canonical_vocab = {x: i + 1 for (i, x) in enumerate(canonical_names)}
        for name in self.names:
            if name not in PAPER:
                raise ValueError(f'Unsupported dataset {name}')
            if isinstance(manifest_path, dict):
                path = Path(require_path(manifest_path.get(name), 'manifest_path.' + name))
            elif manifest_path:
                mp = Path(manifest_path)
                path = mp / name / 'aligned_samples.jsonl' if mp.is_dir() else mp
                if not mp.is_dir() and len(self.names) > 1:
                    raise ValueError('A multi-dataset manifest_path must be a directory or name/path mapping')
            else:
                path = self.root / name / 'aligned_samples.jsonl'
            report = audit_dataset(self.root, name, path, self.sfreq_overrides)
            self.audits[name] = report
            if report['errors'] and (not allow_audited_mismatch):
                raise ValueError(f'{name} fails the paper data audit: ' + '; '.join(report['errors']))
            if report['errors']:
                warnings.warn(f'{name}: audited data mismatches are allowed for diagnostic execution', RuntimeWarning)
            selected = [r for r in iter_index(path) if r['split'] == self.split]
            if max_samples is not None:
                if max_samples <= 0:
                    raise ValueError('max_samples must be positive')
                selected = selected[:max_samples]
            for row in selected:
                row['dataset_name'] = name
                for channel in row['channel_names']:
                    canonical = canonical_channel(channel)
                    if canonical not in self.canonical_vocab:
                        self.canonical_vocab[canonical] = len(self.canonical_vocab) + 1
            self.records.extend(selected)
            keys = {row['sample_key'] for row in selected}
            self.metrics[name] = _JSONLIndex(self.root / name / 'window_metrics.jsonl', keys)
            tp = (targets_path or {}).get(name) if isinstance(targets_path, dict) else targets_path
            self.targets[name] = _JSONLIndex(tp or self.root / name / 'independent_targets.jsonl', keys)
        self.num_channel_ids = max(self.channel_vocab.values(), default=0) + 1 if channel_identity == 'stored' else len(self.canonical_vocab) + 1
        if not self.records:
            raise ValueError('Requested split contains no samples')

    def __len__(self):
        return len(self.records)

    def _file(self, name):
        if self.h5_pid != os.getpid():
            self.close()
            self.h5_pid = os.getpid()
        if name not in self.h5:
            self.h5[name] = h5py.File(self.root / name / 'data_native.h5', 'r')
        return self.h5[name]

    def __getitem__(self, index):
        row = self.records[index]
        (name, key) = (row['dataset_name'], row['sample_key'])
        group = self._file(name)['samples'][key]
        raw = group['signal'][...]
        if list(raw.shape) != row['signal_shape']:
            raise ValueError('H5 signal and index shape disagree')
        if len(raw.shape) != 2 or raw.shape[0] != len(row['channel_names']):
            raise ValueError('H5 signal/channel metadata disagree')
        if int(group.attrs.get('label_id', row['label_id'])) != int(row['label_id']):
            raise ValueError('H5 and manifest labels disagree')
        sfreq = self.sfreq_overrides.get(name, row.get('stored_sfreq_hz', row.get('orig_sfreq_hz')))
        (x, channels_valid) = preprocess_eeg(raw, name, float(sfreq), self.bandpass, self.notch_hz)
        if not channels_valid.any():
            raise ValueError('No finite, non-flat EEG channels are available')
        if self.channel_identity == 'stored':
            ids = group['channel_name_ids'][...].astype(np.int64) if 'channel_name_ids' in group else np.array([self.channel_vocab[c] for c in row['channel_names']])
            expected = np.array([self.channel_vocab[c] for c in row['channel_names']])
            if not np.array_equal(ids, expected):
                raise ValueError('H5 channel IDs disagree with channel_vocab')
        else:
            ids = np.array([self.canonical_vocab[canonical_channel(c)] for c in row['channel_names']])
        (patches, valid, chans, times, discarded) = patchify(x, ids, channels_valid)
        auxiliary = self.metrics[name].get(key)
        if self.auxiliary_policy == 'raw':
            from .physiology import compute_auxiliary_metrics
            provided = self.auxiliary_signals(row) if callable(self.auxiliary_signals) else self.auxiliary_signals.get((name, key), {})
            if not isinstance(provided, dict):
                raise ValueError('The raw auxiliary provider must return a signal mapping')
            auxiliary = compute_auxiliary_metrics(resample_eeg(raw, float(sfreq)), TARGET_SFREQ, eog=provided.get('eog'), ecg=provided.get('ecg'), emg=provided.get('emg'), eog_diff=provided.get('eog_diff'), config=self.physiology_config)
        (numerical, observed, modalities) = descriptor_vector(x, TARGET_SFREQ, auxiliary, self.descriptor_config, self.auxiliary_policy)
        raw_numerical = numerical.copy()
        raw_observed = observed.copy()
        if self.standardizer is not None:
            (numerical, observed) = self.standardizer.transform(numerical, observed)
        label = int(row['label_id'])
        if name == 'TUEV':
            label -= 1
        if not 0 <= label < PAPER[name]['classes']:
            raise ValueError('Label ID is outside the dataset class vocabulary')
        independent = self.targets[name].get(key)
        texts = independent.get('targets', {}) if independent else {}
        if not isinstance(texts, dict) or any((not isinstance(v, str) for v in texts.values())):
            raise ValueError('Independent targets must be a targets mapping of strings')
        texts = dict(texts)
        provenance = independent.get('provenance', 'independent_annotation') if independent else 'label_only'
        if self.smoke_template_targets and (not texts):
            texts = {'summary': descriptor_text(raw_numerical, raw_observed)}
            provenance = 'descriptor_template_smoke_only'
        texts.setdefault('label', str(row.get('label_text', label)))
        target = texts.get('summary', '')
        prompt = 'Use the EEG waveform representation and numerical physiological covariates. ' + QUESTIONS[name]
        return {'eeg': torch.from_numpy(patches.copy()), 'input_mask': torch.from_numpy(valid), 'input_chans': torch.from_numpy(chans), 'input_times': torch.from_numpy(times), 'descriptors': torch.from_numpy(numerical), 'descriptor_mask': torch.from_numpy(observed), 'raw_descriptors': torch.from_numpy(raw_numerical), 'raw_descriptor_mask': torch.from_numpy(raw_observed), 'modality_mask': torch.from_numpy(modalities), 'labels': label, 'prompt': prompt, 'target': target, 'targets': texts, 'label_target': str(row.get('label_text', label)), 'target_provenance': provenance, 'descriptor_text': descriptor_text(raw_numerical, raw_observed), 'sample_id': _hash(f'{name}/{key}'), 'dataset_name': name, 'split': self.split, 'discarded_tail_samples': discarded, 'standardized': self.standardizer is not None, 'auxiliary_policy': self.auxiliary_policy, 'descriptor_extraction_sha256': self.extraction_fingerprint}

    def fit_standardizer(self):
        if self.split != 'train' or self.standardizer is not None:
            raise ValueError('Fit from an unstandardized training dataset only')
        scaler = DescriptorStandardizer().fit((self[i] for i in range(len(self))), split=self.split, extractor_fingerprint=self.extraction_fingerprint)
        self.standardizer = scaler
        return scaler

    def close(self):
        for handle in self.h5.values():
            handle.close()
        self.h5 = {}
        for index in list(self.metrics.values()) + list(self.targets.values()):
            index.close()

    def __getstate__(self):
        return {**self.__dict__, 'h5': {}, 'h5_pid': None}

    def __del__(self):
        if hasattr(self, 'h5'):
            self.close()
Dataset = H5EEGDataset

def collate(samples):
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    maximum = max((len(row['eeg']) for row in samples))
    batch = {'eeg': torch.zeros(len(samples), maximum, PATCH_SIZE), 'input_mask': torch.zeros(len(samples), maximum, dtype=torch.bool), 'input_chans': torch.zeros(len(samples), maximum, dtype=torch.long), 'input_times': torch.zeros(len(samples), maximum, dtype=torch.long)}
    for (i, sample) in enumerate(samples):
        count = len(sample['eeg'])
        for key in ('eeg', 'input_mask', 'input_chans', 'input_times'):
            batch[key][i, :count] = sample[key]
    for key in ('descriptors', 'descriptor_mask', 'modality_mask', 'raw_descriptors', 'raw_descriptor_mask'):
        batch[key] = torch.stack([sample[key] for sample in samples])
    batch['labels'] = torch.tensor([sample['labels'] for sample in samples], dtype=torch.long)
    for key in ('prompt', 'target', 'targets', 'label_target', 'target_provenance', 'descriptor_text', 'dataset_name', 'split', 'discarded_tail_samples', 'standardized', 'auxiliary_policy', 'descriptor_extraction_sha256'):
        batch[key] = [sample[key] for sample in samples]
    batch['sample_ids'] = [sample['sample_id'] for sample in samples]
    return batch
collate_fn = collate

def build_paper_index(root, destination, names=DATASET_NAMES, seed=0, seedv_subject_sets=None, sfreq_overrides=None, mumtaz_test_subjects=None, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    destination = Path(destination)
    root = Path(require_path(root, 'data_root'))
    reports = {}

    def ranked(values, namespace):
        return sorted(values, key=lambda value: _hash(f'{seed}/{namespace}/{value}'))
    for name in names:
        if name not in PAPER:
            raise ValueError(f'Unsupported dataset {name}')
        original = root / name / 'aligned_samples.jsonl'
        rows = list(iter_index(original))
        source_hash = _file_hash(original)
        choices = {'seed': int(seed), 'source_manifest_sha256': source_hash, 'validation_fraction': validation_fraction}
        if name == 'TUEV':
            native_patients = {}
            with original.open() as handle:
                for line in handle:
                    source = json.loads(line)
                    patient = Path(source.get('raw_path', '')).parent.name
                    if not re.fullmatch('(?:[a-z]{8}|\\d{3})', patient):
                        raise ValueError('TUEV raw metadata does not identify a verified patient directory')
                    native_patients[source['sample_key']] = patient
            for row in rows:
                row['subject_id'] = native_patients[row['sample_key']]
            choices['partition_provenance'] = 'preserved native official split; patient IDs corrected from verified enclosing EDF patient directories'
            choices['patient_identity_provenance'] = 'native EDF enclosing patient directory; source paths omitted'
        elif name == 'SEEDV':
            if not seedv_subject_sets or set(seedv_subject_sets) != {'train', 'val', 'test'}:
                raise ValueError('Explicit SEED-V train/val/test subject sets are required')
            sets = {s: {str(x).casefold() for x in values} for (s, values) in seedv_subject_sets.items()}
            if any((len(values) != 5 or '7' in values for values in sets.values())):
                raise ValueError('SEED-V needs five subjects per split, excluding original subject 7')
            if any((sets[a] & sets[b] for (a, b) in (('train', 'val'), ('train', 'test'), ('val', 'test')))):
                raise ValueError('SEED-V subject sets overlap')
            present = {normalized_subject(name, row['subject_id']) for row in rows}
            if set().union(*sets.values()) != present - {'7'}:
                raise ValueError('SEED-V retained subject sets do not cover the available retained corpus')
            rows = [r for r in rows if normalized_subject(name, r['subject_id']) != '7']
            for row in rows:
                subject = normalized_subject(name, row['subject_id'])
                row['split'] = next((s for (s, subjects) in sets.items() if subject in subjects))
            choices['subject_sets'] = {s: sorted(v) for (s, v) in sets.items()}
            choices['partition_provenance'] = 'explicit_subject_sets; exact paper IDs unavailable'
        elif name == 'BCICIV_2a':
            training = defaultdict(list)
            for row in rows:
                match = re.match('A\\d+(T|E)', str(row.get('recording_id', '')), flags=re.I)
                if not match:
                    raise ValueError('BCIC recording IDs do not identify native T/E sessions')
                if match[1].upper() == 'E':
                    row['split'] = 'test'
                else:
                    row['split'] = 'train'
                    training[normalized_subject(name, row['subject_id'])].append(row)
            for (subject, subject_rows) in training.items():
                n = max(1, int(round(validation_fraction * len(subject_rows))))
                keys = set(ranked([r['sample_key'] for r in subject_rows], name + '/' + subject)[:n])
                for row in subject_rows:
                    if row['sample_key'] in keys:
                        row['split'] = 'val'
            choices['partition_provenance'] = 'native T/E sessions; seeded per-subject training-trial validation'
        elif name in ('TUAB', 'mumtaz'):
            test = {normalized_subject(name, r['subject_id']) for r in rows if r['split'] == 'test'}
            if name == 'mumtaz' and mumtaz_test_subjects is not None:
                test = {normalized_subject(name, x) for x in mumtaz_test_subjects}
            available = {normalized_subject(name, r['subject_id']) for r in rows}
            if not test <= available or (name == 'mumtaz' and len(test) != 11):
                raise ValueError('Invalid test subject set')
            pool = available - test
            nval = max(1, int(round(validation_fraction * len(pool))))
            validation = set(ranked(pool, name)[:nval])
            for row in rows:
                subject = normalized_subject(name, row['subject_id'])
                row['split'] = 'test' if subject in test else 'val' if subject in validation else 'train'
            choices['test_subject_hashes'] = sorted((_hash(x) for x in test))
            choices['validation_subject_hashes'] = sorted((_hash(x) for x in validation))
            choices['partition_provenance'] = 'preserved test patients; seeded disjoint validation patients'
        elif name in ('mental-arithmetic', 'SHU-MI'):
            for row in rows:
                number = int(re.search('(\\d+)$', str(row['subject_id']))[1])
                if name == 'mental-arithmetic':
                    row['split'] = 'train' if number <= 27 else 'val' if number <= 31 else 'test'
                else:
                    row['split'] = 'train' if number <= 15 else 'val' if number <= 20 else 'test'
            choices['partition_provenance'] = 'explicit subject IDs in Appendix D'
        else:
            choices['partition_provenance'] = 'preserved existing official-source partition; verify provenance independently'
        override = (sfreq_overrides or {}).get(name)
        if override is not None:
            if not np.isfinite(override) or override <= 0:
                raise ValueError('A stored sampling-rate override must be positive')
            for row in rows:
                row['stored_sfreq_hz'] = float(override)
            choices['stored_sfreq_override_hz'] = float(override)
            choices['override_provenance'] = 'caller-audited source metadata; retain original rate and duration for inspection'
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'aligned_samples.jsonl'
        with path.open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(',', ':')) + '\n')
        report = audit_dataset(root, name, manifest_path=path, sfreq_overrides=sfreq_overrides)
        report['implementation_choices'] = choices
        (folder / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
        reports[name] = report
    return reports

def require_path(value, field):
    if value is None or (isinstance(value, str) and (not value.strip())):
        raise ValueError(f"Set '{field}' before running this command")
    return value

def require_paths(config, *fields):
    for field in fields:
        value = config
        for part in field.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        require_path(value, field)
