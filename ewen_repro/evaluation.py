from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping
from .metrics import FIELDS, canonical_sha256, direct_claim_metrics, field_accuracy, parse_band, parse_direct_claims, parse_label, parse_withheld, summary_rouge_l, withheld_metrics, _sidecar

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while (data := handle.read(1 << 20)):
            digest.update(data)
    return digest.hexdigest()

def _jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Evaluation JSONL records must be objects')
            yield row

def _prediction_rows(predictions_path, predictions_metadata_path=None):
    path = Path(predictions_path)
    records = {}
    (diagnostics, conditions, datasets) = (set(), set(), set())
    for row in _jsonl(path):
        (sid, field, text) = (row.get('sample_id'), row.get('field'), row.get('text'))
        if not isinstance(sid, str) or not sid or field not in FIELDS or (not isinstance(text, str)) or (type(row.get('diagnostic')) is not bool):
            raise ValueError('Predictions require sample_id, declared field, text and explicit boolean diagnostic')
        if not isinstance(row.get('dataset'), str) or not row['dataset']:
            raise ValueError('Predictions require an explicit dataset identity')
        key = (sid, field)
        if key in records:
            raise ValueError('Duplicate sample/field predictions')
        records[key] = row
        diagnostics.add(row['diagnostic'])
        datasets.add(row['dataset'])
        if 'condition' in row:
            if row['condition'] not in {'full', 'waveform_only', 'descriptor_only', 'prompt_only'}:
                raise ValueError('Unknown prediction information condition')
            conditions.add(row['condition'])
    if not records or len(diagnostics) != 1 or len(conditions) > 1:
        raise ValueError('Prediction set must be nonempty with a consistent diagnostic status and information condition')
    if conditions and any(('condition' not in row for row in records.values())):
        raise ValueError('Information-condition metadata must be present consistently')
    diagnostic = next(iter(diagnostics))
    prediction_sha = _sha(path)
    metadata_path = Path(predictions_metadata_path) if predictions_metadata_path else path.with_suffix('.metadata.json')
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        stored = metadata.get('prediction_sha256', metadata.get('predictions_sha256', metadata.get('source_sha256')))
        if stored != prediction_sha:
            raise ValueError('Prediction metadata hash does not identify the supplied prediction file')
        if metadata.get('diagnostic') is not diagnostic:
            raise ValueError('Prediction rows and metadata disagree on diagnostic status')
        if metadata.get('split') != 'test':
            raise ValueError('Paper field/summary scoring requires test predictions')
        if set(metadata.get('datasets', [])) != datasets:
            raise ValueError('Prediction rows and metadata dataset identities differ')
        metadata_condition = metadata.get('condition', metadata.get('information_condition'))
        if conditions and metadata_condition != next(iter(conditions)):
            raise ValueError('Prediction rows and metadata information conditions differ')
        metadata_fields = metadata.get('fields', [metadata.get('field')])
        if set(metadata_fields) != {field for (_, field) in records}:
            raise ValueError('Prediction rows and metadata task fields differ')
    elif predictions_metadata_path:
        raise ValueError('Requested prediction metadata file is absent')
    provenance = {'diagnostic': diagnostic, 'artifact_validity': 'diagnostic_model_outputs' if diagnostic else 'scored_supplied_model_outputs', 'information_condition': next(iter(conditions)) if conditions else None, 'prediction_metadata_verified': metadata is not None, 'checkpoint_sha256': metadata.get('checkpoint_sha256') if metadata else None, 'resource_identity_sha256': canonical_sha256(metadata['resource_identity']) if metadata and 'resource_identity' in metadata else None, 'source_hashes': {'predictions': prediction_sha}, 'prediction_count': len(records)}
    if metadata:
        provenance['source_hashes']['prediction_metadata'] = _sha(metadata_path)
    return (records, provenance)

def _test_targets(path):
    targets = {}
    for row in _jsonl(path):
        if row.get('split') != 'test':
            continue
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid or sid in targets:
            raise ValueError('Test target IDs must be nonempty and unique')
        if row.get('provenance') not in {'independent_waveform_measurements', 'independent_annotation'}:
            raise ValueError('Test targets require independent waveform or annotation provenance')
        name = row.get('dataset', row.get('dataset_name'))
        if not isinstance(name, str) or not name:
            raise ValueError('Test targets require an explicit dataset identity')
        if not isinstance(row.get('targets'), Mapping) or not isinstance(row['targets'].get('summary'), str):
            raise ValueError('Test target requires its independent summary reference')
        targets[sid] = {**row, 'dataset': name}
    if not targets:
        raise ValueError('No independent test targets')
    return targets

def _select_targets(targets, rows, provenance, sample_ids_path):
    if not sample_ids_path:
        provenance['reference_scope'] = {'mode': 'complete_frozen_test', 'complete_test_count': len(targets), 'selected_test_count': len(targets)}
        return targets
    if provenance['diagnostic'] is not True:
        raise ValueError('Explicit test subsets are restricted to diagnostic predictions')
    identifiers = json.loads(Path(sample_ids_path).read_text())
    if not isinstance(identifiers, list) or not identifiers or any((not isinstance(sid, str) or not sid for sid in identifiers)) or (len(set(identifiers)) != len(identifiers)):
        raise ValueError('Diagnostic sample selection requires a nonempty unique JSON list of sample IDs')
    selected = set(identifiers)
    if selected != {sid for (sid, _) in rows} or not selected.issubset(targets):
        raise ValueError('Diagnostic selection must equal predicted IDs and belong to frozen test targets')
    provenance['source_hashes']['sample_selection'] = _sha(sample_ids_path)
    provenance['reference_scope'] = {'mode': 'explicit_diagnostic_subset', 'complete_test_count': len(targets), 'selected_test_count': len(selected), 'selected_sample_ids_sha256': canonical_sha256(sorted(selected))}
    return {sid: targets[sid] for sid in sorted(selected)}

def _select_sidecar(sidecar, kind, complete_ids, selected_ids, provenance, name):
    (records, _, fingerprint) = _sidecar(sidecar, kind)
    if set(records) != complete_ids:
        raise ValueError('Frozen sidecars must match the complete independent test target set')
    if selected_ids == complete_ids:
        return sidecar
    provenance['reference_scope'][name + '_complete_reference_sha256'] = fingerprint
    return {**sidecar, 'records': [records[sid] for sid in sorted(selected_ids)]}

def score_summaries(predictions_path, targets_path, direct_reference_path, withheld_reference_path, *, predictions_metadata_path=None, sample_ids_path=None):
    from .grounding import clause_bank
    (rows, provenance) = _prediction_rows(predictions_path, predictions_metadata_path)
    if any((field != 'summary' for (_, field) in rows)):
        raise ValueError('Summary scoring requires a summary-only prediction set')
    targets = _test_targets(targets_path)
    complete_ids = set(targets)
    targets = _select_targets(targets, rows, provenance, sample_ids_path)
    predicted = {sid: row['text'] for ((sid, _), row) in rows.items()}
    if set(predicted) != set(targets):
        raise ValueError('Predictions must cover the exact frozen test reference set')
    if any((row['dataset'] != targets[sid]['dataset'] for ((sid, _), row) in rows.items())):
        raise ValueError('Prediction and target dataset identities differ')
    reference = {sid: row['targets']['summary'] for (sid, row) in targets.items()}
    direct_sidecar = json.loads(Path(direct_reference_path).read_text())
    withheld_sidecar = json.loads(Path(withheld_reference_path).read_text())
    direct_sidecar = _select_sidecar(direct_sidecar, 'direct_numeric', complete_ids, set(targets), provenance, 'direct')
    withheld_sidecar = _select_sidecar(withheld_sidecar, 'withheld_states', complete_ids, set(targets), provenance, 'withheld')
    bank = withheld_sidecar.get('parser_clause_bank', clause_bank())
    declared = withheld_sidecar.get('attributes', [])
    if not declared or any((attribute not in bank for attribute in declared)):
        raise ValueError('Withheld reference attributes are absent from the fixed clause bank')
    bank = {attribute: bank[attribute] for attribute in declared}
    provenance['source_hashes'].update(targets=_sha(targets_path), direct_reference=_sha(direct_reference_path), withheld_reference=_sha(withheld_reference_path))
    return {**provenance, 'summary_rouge_l': summary_rouge_l(predicted, reference), 'direct_f1': direct_claim_metrics({sid: parse_direct_claims(text) for (sid, text) in predicted.items()}, direct_sidecar), 'withheld_f1': withheld_metrics({sid: parse_withheld(text, bank) for (sid, text) in predicted.items()}, withheld_sidecar), 'parser_clause_bank_sha256': canonical_sha256(bank), 'scoring_scope': 'Complete supplied autoregressive summary outputs; exact reconstruction clause bank'}

def _field_parser(path):
    config = json.loads(Path(path).read_text())
    if config.get('kind') != 'field_parser_config' or config.get('frozen') is not True:
        raise ValueError('Field evaluation requires a frozen explicit parser config')
    (labels, relations) = (config.get('label_sets'), config.get('relation_clause_bank'))
    if not isinstance(labels, Mapping) or not labels or (not isinstance(relations, Mapping)):
        raise ValueError('Parser config requires dataset label sets and relation clause banks')
    if set(labels) != set(relations):
        raise ValueError('Label and relation parser dataset sets differ')
    for name in labels:
        if not isinstance(labels[name], list) or not labels[name] or len(set(labels[name])) != len(labels[name]):
            raise ValueError('Declared dataset labels must be nonempty and unique')
        bank = relations[name]
        if not isinstance(bank, Mapping) or not bank:
            raise ValueError('Every dataset requires an explicit relation-state clause bank')
        for (state, patterns) in bank.items():
            if not isinstance(state, str) or not isinstance(patterns, list) or (not patterns):
                raise ValueError('Relation-state patterns must be explicit nonempty lists')
            for pattern in patterns:
                re.compile(pattern)
    return config

def _parse_relation(text, bank):
    normalized = ' '.join(text.casefold().split())
    states = {state for (state, patterns) in bank.items() if any((re.search(pattern, normalized) for pattern in patterns))}
    return next(iter(states)) if len(states) == 1 else None

def score_fields(predictions_path, targets_path, parser_config_path, *, predictions_metadata_path=None, sample_ids_path=None):
    (rows, provenance) = _prediction_rows(predictions_path, predictions_metadata_path)
    (targets, parser) = (_test_targets(targets_path), _field_parser(parser_config_path))
    targets = _select_targets(targets, rows, provenance, sample_ids_path)
    expected = {(sid, field) for sid in targets for field in FIELDS}
    if set(rows) != expected:
        raise ValueError('Four-field evaluation requires all four independent task outputs for every test window')
    (parsed, refs) = ({}, {})
    for (sid, target) in targets.items():
        name = target['dataset']
        if name not in parser['label_sets']:
            raise ValueError('Target dataset absent from frozen parser')
        fields = target.get('fields')
        if not isinstance(fields, Mapping) or set(fields) != set(FIELDS):
            raise ValueError('Independent field target requires label, relation, band, summary')
        if fields['label'] not in parser['label_sets'][name] or fields['band'] not in ('delta', 'theta', 'alpha', 'beta', 'gamma') or fields['relation'] not in parser['relation_clause_bank'][name] or (fields['summary'] != {f: fields[f] for f in ('label', 'band', 'relation')}):
            raise ValueError('Field reference values are outside the frozen grammar or summary schema')
        if any((rows[sid, f]['dataset'] != name for f in FIELDS)):
            raise ValueError('Prediction and target dataset identities differ')
        (labels, relation_bank) = (parser['label_sets'][name], parser['relation_clause_bank'][name])
        summary = rows[sid, 'summary']['text']
        parsed[sid] = {'label': parse_label(rows[sid, 'label']['text'], labels), 'band': parse_band(rows[sid, 'band']['text']), 'relation': _parse_relation(rows[sid, 'relation']['text'], relation_bank), 'summary': {'label': parse_label(summary, labels), 'band': parse_band(summary), 'relation': _parse_relation(summary, relation_bank)}}
        refs[sid] = dict(fields)
    provenance['source_hashes'].update(targets=_sha(targets_path), field_parser=_sha(parser_config_path))
    return {**provenance, 'field_accuracy': field_accuracy(parsed, refs), 'paper_exact_aggregation': True, 'paper_exact_relation_targets': parser.get('paper_exact_relation_targets', False), 'parser_basis': parser.get('basis', 'Explicit frozen caller-supplied field grammar')}
RELATION_PATTERNS = {'insufficient evidence': ['\\bauxiliary physiological relation assessment has insufficient evidence\\b', '\\bavailable auxiliary coupling descriptors provide insufficient evidence\\b', '\\b(?:eog|ecg|emg|auxiliary (?:physiological )?(?:measurements|signals?|relations?|coupling))[^.!?]*\\b(?:unavailable|insufficient evidence|not (?:available|recorded)|absent)\\b'], 'numerical evidence without unspecified categorical bins': ['\\bobserved coupling descriptors\\b', '\\b(?:eog|ecg|emg)[-\\s]*eeg\\s+(?:maximum lagged correlation|regression gain|hep amplitude|plv|beta coherence|phase-slope delay)\\s+(?:is|=|:)\\s*-?(?:\\d+(?:\\.\\d+)?|\\.\\d+)\\b']}

def build_field_parser(target_paths, output_path):
    if isinstance(target_paths, (str, Path)):
        target_paths = [target_paths]
    (labels, states) = (defaultdict(set), defaultdict(set))
    (digest, count) = (hashlib.sha256(), 0)
    for path in target_paths:
        for row in _jsonl(path):
            if row.get('split') != 'train':
                continue
            if row.get('provenance') not in {'independent_waveform_measurements', 'independent_annotation'}:
                raise ValueError('Parser training rows need independent target provenance')
            (name, fields) = (row.get('dataset', row.get('dataset_name')), row.get('fields', {}))
            if not isinstance(name, str) or not name or (not isinstance(fields, Mapping)) or (not isinstance(fields.get('label'), str)) or (not fields['label']) or (fields.get('relation') not in RELATION_PATTERNS):
                raise ValueError('Training field schema is outside the explicit reconstruction grammar')
            labels[name].add(fields['label'])
            states[name].add(fields['relation'])
            digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode() + b'\n')
            count += 1
    if not count:
        raise ValueError('No training targets available; test targets cannot fit parser vocabularies')
    result = {'kind': 'field_parser_config', 'frozen': True, 'fit_source_split': 'train', 'training_target_count': count, 'training_rows_sha256': digest.hexdigest(), 'label_sets': {name: sorted(values) for (name, values) in labels.items()}, 'relation_clause_bank': {name: {state: RELATION_PATTERNS[state] for state in sorted(values)} for (name, values) in states.items()}, 'paper_exact_relation_targets': False, 'basis': 'Train-observed reconstruction field categories with fixed explicit clause patterns; the manuscript relation bank is unavailable'}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(destination)
    return result
