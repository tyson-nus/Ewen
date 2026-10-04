from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
STATES = ('negative', 'neutral', 'positive')
FIELDS = ('label', 'relation', 'band', 'summary')
BANDS = ('delta', 'theta', 'alpha', 'beta', 'gamma')
import math
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np

def _result(metrics: dict, *, valid: bool=True, reasons: Sequence[str]=(), evidence: dict | None=None, **details: Any) -> dict:
    return {'validity': 'valid' if valid else 'invalid', 'artifact_validity': 'computed_from_supplied_inputs', 'metrics': metrics, 'reasons': list(reasons), 'denominator_evidence': evidence or {}, **details}

def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number, not a boolean')
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(v):
        raise ValueError(f'{name} must be finite')
    return v

def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)

def _binary_auc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    positives = int(y.sum())
    negatives = len(y) - positives
    return float((ranks[y.astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))

def _binary_pr_area(y: np.ndarray, scores: np.ndarray, method: str) -> float:
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    sorted_y = y[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(sorted_y)[endpoints]
    precision = tp / (endpoints + 1)
    recall = tp / int(y.sum())
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    delta = np.diff(recall)
    if method == 'average_precision':
        return float(np.sum(delta * precision[1:]))
    return float(np.sum(delta * (precision[:-1] + precision[1:]) / 2))

def classification_metrics(y_true: Sequence[int], probabilities: Any, n_classes: int, *, pr_auc_method: str='trapezoid') -> dict:
    k = _positive_int(n_classes, 'n_classes')
    if k < 2:
        raise ValueError('n_classes must be at least two')
    if pr_auc_method not in {'trapezoid', 'average_precision'}:
        raise ValueError('pr_auc_method must be trapezoid or average_precision')
    original_y = np.asarray(y_true)
    if original_y.ndim != 1 or original_y.dtype.kind not in 'iu':
        raise ValueError('y_true must be a one-dimensional integer class array')
    y = original_y.astype(np.int64, copy=False)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), k):
        raise ValueError(f'probabilities must have shape [N,{k}]')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('probabilities must be finite and in [0,1]')
    if len(y) and (not np.allclose(p.sum(axis=1), 1, rtol=1e-05, atol=1e-07)):
        raise ValueError('probability rows must sum to one')
    if np.any(y < 0) or np.any(y >= k):
        raise ValueError('y_true contains an undeclared class')
    support = np.bincount(y, minlength=k)
    evidence = {'sample_count': len(y), 'n_classes': k, 'class_order': list(range(k)), 'class_support': support.tolist()}
    names = ('balanced_accuracy', 'roc_auc', 'pr_auc') if k == 2 else ('balanced_accuracy', 'cohen_kappa', 'weighted_f1')
    missing = np.flatnonzero(support == 0).tolist()
    convention = {'pr_auc_method': pr_auc_method, 'paper_pr_numerical_rule': 'unspecified; paper defines integral precision d(recall)', 'positive_class': 1 if k == 2 else None}
    if not len(y) or missing:
        return _result(dict.fromkeys(names), valid=False, reasons=[f'Ground truth has no support for declared classes {missing}'], evidence=evidence, convention=convention)
    predicted = np.argmax(p, axis=1)
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    tp = np.diag(confusion)
    predicted_support = confusion.sum(axis=0)
    recalls = tp / support
    scores = {'balanced_accuracy': float(recalls.mean())}
    if k == 2:
        scores['roc_auc'] = _binary_auc(y, p[:, 1])
        scores['pr_auc'] = _binary_pr_area(y, p[:, 1], pr_auc_method)
    else:
        observed = float(tp.sum() / len(y))
        expected = float(np.dot(support / len(y), predicted_support / len(y)))
        scores['cohen_kappa'] = (observed - expected) / (1 - expected)
        denominator = support + predicted_support
        per_class_f1 = np.divide(2 * tp, denominator, out=np.zeros(k), where=denominator != 0)
        scores['weighted_f1'] = float(np.dot(support / len(y), per_class_f1))
    evidence.update({'predicted_support': predicted_support.tolist(), 'confusion_matrix': confusion.tolist()})
    return _result(scores, evidence=evidence, convention=convention)

def summarize_runs(values: Sequence[float]) -> dict:
    vals = [_finite_number(v, 'run score') for v in values]
    if not vals:
        return _result({'mean': None, 'sample_std': None}, valid=False, reasons=['No completed runs'], evidence={'run_count': 0})
    return _result({'mean': float(np.mean(vals)), 'sample_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None}, evidence={'run_count': len(vals)}, ddof=1, sample_std_defined=len(vals) > 1)

def dataset_macro_runs(dataset_scores: Mapping[str, Sequence[float]]) -> dict:
    if not dataset_scores:
        raise ValueError('Dataset-macro aggregation requires named datasets')
    lengths = {len(v) for v in dataset_scores.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError('Every dataset requires same ordered nonempty run list')
    arrays = np.array([[_finite_number(x, 'dataset run score') for x in v] for v in dataset_scores.values()])
    run_macros = arrays.mean(axis=0).tolist()
    result = summarize_runs(run_macros)
    result['macro_run_scores'] = run_macros
    result['dataset_weighting'] = 'equal weight within each run before across-run statistics'
    return result

def normalize_label_text(text: str) -> str:
    return ' '.join(str(text).casefold().split())

def parse_label(text: str, label_set: Sequence[str]) -> str | None:
    normalized = normalize_label_text(text)
    labels = {normalize_label_text(label): label for label in label_set}
    if len(labels) != len(label_set) or any((not x for x in labels)):
        raise ValueError('Label set must have distinct nonempty normalized labels')
    matches = [(len(label), original) for (label, original) in labels.items() if re.search('(?<!\\w)' + re.escape(label) + '(?!\\w)', normalized)]
    if not matches:
        return None
    longest = max((length for (length, _) in matches))
    winners = [original for (length, original) in matches if length == longest]
    return winners[0] if len(winners) == 1 else None

def parse_band(text: str) -> str | None:
    normalized = text.casefold()
    aliases = '|'.join(BANDS)
    patterns = ('\\bdominant(?:\\s+eeg)?(?:\\s+(?:spectral\\s+)?band)?(?:\\s+is|\\s*:|\\s*=)?\\s+(' + aliases + ')\\b', '\\b(' + aliases + ')\\s*-\\s*dominant\\b', '\\b(' + aliases + ')\\s+(?:band\\s+)?(?:is\\s+)?dominant\\b')
    explicit = {match.group(1) for pattern in patterns for match in re.finditer(pattern, normalized)}
    if explicit:
        return next(iter(explicit)) if len(explicit) == 1 else None
    found = {band for band in BANDS if re.search('\\b' + band + '\\b', text.casefold())}
    return next(iter(found)) if len(found) == 1 else None

def field_accuracy(predictions: Mapping[str, Mapping[str, Any]], references: Mapping[str, Mapping[str, Any]]) -> dict:
    if not references:
        return _result(dict.fromkeys([*FIELDS, 'total_field_accuracy']), valid=False, reasons=['No reference windows'], evidence={'sample_count': 0})
    if set(predictions) != set(references):
        raise ValueError('Prediction and reference sample IDs must match exactly')
    counts = Counter()
    for (sid, ref) in references.items():
        if set(ref) != set(FIELDS) or any((v is None for v in ref.values())):
            raise ValueError(f'All four reference tasks require explicit target values: {sid}')
        pred = predictions[sid]
        if set(pred) - set(FIELDS):
            raise ValueError(f'Unknown predicted field: {sid}')
        for field in FIELDS:
            if field == 'summary' and isinstance(ref[field], Mapping):
                if set(ref[field]) != {'label', 'band', 'relation'}:
                    raise ValueError('Structured summary reference requires label, band, relation')
                value = pred.get(field)
                correct = isinstance(value, Mapping) and all((value.get(f) == ref[field][f] for f in ('label', 'band', 'relation')))
            else:
                correct = pred.get(field) == ref[field]
            counts[field] += bool(correct)
    n = len(references)
    scores = {field: counts[field] / n for field in FIELDS}
    scores['total_field_accuracy'] = sum(scores.values()) / len(FIELDS)
    return _result(scores, evidence={'sample_count': n, 'correct_per_task': dict(counts)}, convention='Exact equality after documented external parsing; no score-side aliases')

def rouge_l(candidate: str, reference: str) -> float:
    token_pattern = '\\d+\\.\\d+|[a-z0-9]+'
    x = re.findall(token_pattern, str(candidate).lower())
    y = re.findall(token_pattern, str(reference).lower())
    if not x or not y:
        return 0.0
    if len(x) < len(y):
        (x, y) = (y, x)
    previous = [0] * (len(y) + 1)
    for a in x:
        current = [0]
        for (j, b) in enumerate(y, 1):
            current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    return float(2 * lcs / (len(x) + len(y)))

def summary_rouge_l(predictions: Mapping[str, str], references: Mapping[str, str]) -> dict:
    if set(predictions) != set(references):
        raise ValueError('Summary sample IDs must match exactly')
    if not references:
        return _result({'summary_rouge_l': None}, valid=False, reasons=['No summaries'], evidence={'sample_count': 0})
    values = {sid: rouge_l(predictions[sid], reference) for (sid, reference) in references.items()}
    return _result({'summary_rouge_l': float(np.mean(list(values.values())))}, evidence={'sample_count': len(values)}, sample_scores=values)

def canonical_sha256(data: Any) -> str:
    content = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def _sidecar(sidecar: Mapping[str, Any], kind: str) -> tuple[dict, list[str], str]:
    if sidecar.get('kind') != kind:
        raise ValueError(f'Reference sidecar kind must be {kind}')
    if sidecar.get('frozen') is not True or sidecar.get('independent') is not True:
        raise ValueError('References require frozen=True and independent=True')
    if sidecar.get('split') != 'test':
        raise ValueError('Evaluation reference sidecar must identify the test split')
    attributes = sidecar.get('attributes')
    if not isinstance(attributes, list) or not attributes or len(set(attributes)) != len(attributes) or any((not isinstance(a, str) or not a for a in attributes)):
        raise ValueError('Reference sidecar requires distinct explicit attributes')
    records = {}
    for row in sidecar.get('records', []):
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid or sid in records:
            raise ValueError('Reference sample IDs must be nonempty and unique')
        (values, available) = (row.get('values'), row.get('available'))
        if not isinstance(values, Mapping) or not isinstance(available, Mapping):
            raise ValueError('Each reference needs values and an availability mask')
        if set(available) != set(attributes) or any((type(x) is not bool for x in available.values())):
            raise ValueError('Availability must provide a boolean for every declared attribute')
        if set(values) - set(attributes):
            raise ValueError('Undeclared reference target')
        for a in attributes:
            if available[a] and (a not in values or values[a] is None):
                raise ValueError(f'Observable target has no independent value: {sid}/{a}')
        records[sid] = row
    if not records:
        raise ValueError('Frozen reference sidecar has no records')
    fingerprint = canonical_sha256(sidecar)
    return (records, attributes, fingerprint)

def direct_claim_metrics(predictions: Mapping[str, Sequence[Mapping[str, Any]]], frozen_reference_sidecar: Mapping[str, Any], *, tolerance: float=0.005) -> dict:
    if _finite_number(tolerance, 'tolerance') != 0.005:
        raise ValueError('Paper Direct F1 uses fixed absolute tolerance 0.005')
    (refs, attrs, fingerprint) = _sidecar(frozen_reference_sidecar, 'direct_numeric')
    if set(attrs) - set(BANDS):
        raise ValueError('Paper Direct F1 targets relative powers of the five canonical bands')
    if set(predictions) != set(refs):
        raise ValueError('Direct prediction IDs must match reference windows exactly')
    totals = Counter(tp=0, fp=0, fn=0, available=0, unavailable=0, claims=0)
    per_window = {}
    for (sid, ref) in refs.items():
        target = {a: _finite_number(ref['values'][a], f'{sid}/{a}') for a in attrs if ref['available'][a]}
        if any((v < 0 or v > 1 for v in target.values())):
            raise ValueError('Relative power reference values must be in [0,1]')
        matched = set()
        local = Counter(tp=0, fp=0, fn=0, claims=0)
        for claim in predictions[sid]:
            if set(claim) != {'attribute', 'value'}:
                raise ValueError('Direct claim requires exactly attribute and value')
            a = claim['attribute']
            if a not in BANDS:
                raise ValueError(f'Unknown numeric band claim: {a}')
            value = _finite_number(claim['value'], f'{sid} claim')
            local['claims'] += 1
            if a in target and a not in matched and (abs(value - target[a]) <= tolerance + 1e-12):
                local['tp'] += 1
                matched.add(a)
            else:
                local['fp'] += 1
        local['fn'] = len(target) - len(matched)
        local['available'] = len(target)
        local['unavailable'] = len(attrs) - len(target)
        totals.update(local)
        per_window[sid] = dict(local)
    denominator = 2 * totals['tp'] + totals['fp'] + totals['fn']
    score = 2 * totals['tp'] / denominator if denominator else 0.0
    return _result({'direct_f1': score}, evidence={**dict(totals), 'f1_denominator': denominator, 'sample_count': len(refs)}, reference_sha256=fingerprint, per_window=per_window, tolerance=tolerance, claim_matching='One reference per band; match a correct claim once; all other claims FP; unmatched references FN')

def parse_direct_claims(text: str) -> list[dict]:
    pattern = '\\b(delta|theta|alpha|beta|gamma)\\b(?:\\s+band)?\\s+(?:relative\\s+)?power\\s*(?:is|of|=|:)?\\s*(-?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s*(%)?'
    claims = []
    for match in re.finditer(pattern, text.casefold()):
        v = float(match.group(2)) / (100 if match.group(3) else 1)
        claims.append({'attribute': match.group(1), 'value': v})
    return claims

def parse_withheld(text: str, clause_bank: Mapping[str, Mapping[str, Sequence[str]]]) -> dict:
    normalized = ' '.join(text.casefold().split())
    parsed = {}
    for (attribute, patterns) in clause_bank.items():
        if set(patterns) != set(STATES):
            raise ValueError('Each attribute clause bank requires all three states')
        found = {state for state in STATES if any((re.search(pattern, normalized) for pattern in patterns[state]))}
        parsed[attribute] = next(iter(found)) if len(found) == 1 else None
    return parsed

def withheld_metrics(predictions: Mapping[str, Mapping[str, Any]], frozen_reference_sidecar: Mapping[str, Any]) -> dict:
    (refs, attrs, fingerprint) = _sidecar(frozen_reference_sidecar, 'withheld_states')
    if set(predictions) != set(refs):
        raise ValueError('Withheld prediction IDs must match reference windows exactly')
    counts = {a: {s: Counter(tp=0, fp=0, fn=0, support=0) for s in STATES} for a in attrs}
    available = Counter()
    unavailable = Counter()
    missing = Counter()
    for (sid, ref) in refs.items():
        pred = predictions[sid]
        if not isinstance(pred, Mapping) or set(pred) - set(attrs):
            raise ValueError(f'Unknown withheld attribute in prediction: {sid}')
        for a in attrs:
            if not ref['available'][a]:
                unavailable[a] += 1
                continue
            truth = ref['values'][a]
            if truth not in STATES:
                raise ValueError(f'Observable reference must use an explicit three-state value: {sid}/{a}')
            available[a] += 1
            counts[a][truth]['support'] += 1
            guess = pred.get(a)
            if not isinstance(guess, str) or guess not in STATES:
                guess = None
            if guess == truth:
                counts[a][truth]['tp'] += 1
            else:
                counts[a][truth]['fn'] += 1
                if guess is None:
                    missing[a] += 1
                else:
                    counts[a][guess]['fp'] += 1
    per_attribute = {}
    for a in attrs:
        if available[a] == 0:
            continue
        class_f1 = []
        for state in STATES:
            row = counts[a][state]
            denominator = 2 * row['tp'] + row['fp'] + row['fn']
            class_f1.append(2 * row['tp'] / denominator if denominator else 0.0)
        per_attribute[a] = float(np.mean(class_f1))
    evidence = {'sample_count': len(refs), 'declared_attributes': attrs, 'observable_attributes': list(per_attribute), 'available_targets': {a: available[a] for a in attrs}, 'unavailable_targets': {a: unavailable[a] for a in attrs}, 'missing_predictions': {a: missing[a] for a in attrs}, 'class_counts': {a: {s: dict(counts[a][s]) for s in STATES} for a in attrs}}
    return _result({'withheld_f1': float(np.mean(list(per_attribute.values()))) if per_attribute else None}, valid=bool(per_attribute), reasons=[] if per_attribute else ['No observable withheld attributes'], evidence=evidence, attribute_f1=per_attribute, reference_sha256=fingerprint)
