"""Evaluation metrics for Ewen (paper §C).

* Closed-vocabulary (§C.1)        — :func:`binary_metrics`, :func:`multiclass_metrics`
* Open-ended (§C.2, Eq. 34–35)    — :func:`field_accuracy`, :func:`total_f_acc`
* BERTScore (§C.2, Eq. 36)        — :func:`bertscore`
* Orthogonal-update geometry      — :func:`leakage_angle` (Eq. 54), :func:`gradient_cosine`
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------- §C.1 metrics


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Paper §C.1: balanced acc / ROC AUC / PR AUC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    out = {"acc_b": _balanced_accuracy(y_true, y_pred)}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = 0.0
        out["pr_auc"] = 0.0
    return out


def multiclass_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Paper §C.1: balanced acc / Cohen's κ / weighted F1."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_score.ndim == 1:
        y_pred = y_score.astype(int)
    else:
        y_pred = y_score.argmax(axis=-1)
    return {
        "acc_b": _balanced_accuracy(y_true, y_pred),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# --------------------------------------------------------- §C.2 open-ended F-Acc


def field_accuracy(pred_fields: Sequence[str], gt_fields: Sequence[str]) -> float:
    """Per-sub-task field accuracy (paper Eq. 34): exact-match rate after normalization."""
    if len(pred_fields) != len(gt_fields):
        raise ValueError("pred_fields and gt_fields must have the same length")
    if not pred_fields:
        return 0.0
    correct = sum(1 for p, g in zip(pred_fields, gt_fields)
                  if str(p).strip().lower() == str(g).strip().lower())
    return correct / len(pred_fields)


def total_f_acc(per_subtask_accuracies: Dict[str, float]) -> float:
    """Headline scalar of paper Eq. 35: unweighted mean of the four sub-task F-Accs."""
    if not per_subtask_accuracies:
        return 0.0
    return float(np.mean(list(per_subtask_accuracies.values())))


# ------------------------------------------------------- §C.2 BERTScore (Eq. 36)


@torch.no_grad()
def _embed_for_bertscore(texts: Sequence[str], tokenizer, model, device,
                          max_length: int = 256):
    """Tokenize, run through a frozen encoder, and return contextual embeddings +
    attention masks. The encoder is expected to live in eval mode on ``device``."""
    tok = tokenizer(list(texts), return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length)
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok, output_hidden_states=False)
    emb = out.last_hidden_state  # (B, L, d)
    return emb, tok["attention_mask"]


@torch.no_grad()
def bertscore(candidates: Sequence[str], references: Sequence[str],
              tokenizer=None, encoder=None, model_name: str = "roberta-large",
              device: str | torch.device = "cpu",
              rescale_baseline: float = 0.85,
              batch_size: int = 16) -> Dict[str, float]:
    """Paper §C.2 / Eq. 36 — F1 of greedy-aligned cosine similarity between contextual
    embeddings of ``candidates`` and ``references``.

    Returns
    -------
    dict
        ``{"bertscore_p": ..., "bertscore_r": ..., "bertscore_f1": ...}`` where the F1
        is the rescaled variant (``(raw − baseline) / (1 − baseline)``), matching the
        ``rescale_with_baseline`` option used in the paper.
    """
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have the same length")
    if not candidates:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    if tokenizer is None or encoder is None:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    precisions: List[float] = []
    recalls: List[float] = []
    for start in range(0, len(candidates), batch_size):
        cand_batch = candidates[start: start + batch_size]
        ref_batch = references[start: start + batch_size]
        cand_emb, cand_mask = _embed_for_bertscore(cand_batch, tokenizer, encoder, device)
        ref_emb, ref_mask = _embed_for_bertscore(ref_batch, tokenizer, encoder, device)
        for c_emb, c_msk, r_emb, r_msk in zip(cand_emb, cand_mask, ref_emb, ref_mask):
            c = F.normalize(c_emb[c_msk.bool()], dim=-1)
            r = F.normalize(r_emb[r_msk.bool()], dim=-1)
            if c.numel() == 0 or r.numel() == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                continue
            sim = c @ r.t()                       # (L_c, L_r)
            precision = sim.max(dim=1).values.mean().item()   # mean over candidate tokens
            recall = sim.max(dim=0).values.mean().item()      # mean over reference tokens
            precisions.append(precision)
            recalls.append(recall)
    p = float(np.mean(precisions))
    r = float(np.mean(recalls))
    f1 = 0.0 if (p + r) <= 0 else 2 * p * r / (p + r)
    if rescale_baseline > 0:
        # Same rescale-with-baseline transform as the paper / the bert-score library.
        f1 = (f1 - rescale_baseline) / (1.0 - rescale_baseline)
    return {"bertscore_p": p, "bertscore_r": r, "bertscore_f1": f1}


def bertscore_split(candidates: Sequence[str],
                    references_owt: Sequence[str],
                    references_dataset: Sequence[str],
                    **kwargs) -> Dict[str, float]:
    """Decomposition used in Figure 5: OWT-BERT / Dataset-BERT / Mix-BERT."""
    owt = bertscore(candidates, references_owt, **kwargs)
    dset = bertscore(candidates, references_dataset, **kwargs)
    mix_refs = list(references_owt) + list(references_dataset)
    mix_cands = list(candidates) + list(candidates)
    mix = bertscore(mix_cands, mix_refs, **kwargs)
    return {
        "owt_bert": owt["bertscore_f1"],
        "dataset_bert": dset["bertscore_f1"],
        "mix_bert": mix["bertscore_f1"],
    }


# ----------------------------------------------------- Orthogonal-update geometry


def leakage_angle(update: torch.Tensor | Sequence[torch.Tensor],
                  lang_grad: torch.Tensor | Sequence[torch.Tensor]) -> float:
    """Paper Eq. 54 — angle between the realised update ``u`` and the language-sensitive
    direction ``g_lang``:

        α_leak = arcsin(|⟨u, g_lang⟩| / (‖u‖ · ‖g_lang‖))     ∈ [0, π/2]

    ``update`` and ``lang_grad`` can be a single tensor or matching per-parameter
    lists (matching the orthogonal-LoRA bookkeeping in :mod:`ewen.model.orthogonal_lora`).
    Returns the angle in radians; multiply by 180/π for degrees.
    """
    if isinstance(update, torch.Tensor) and isinstance(lang_grad, torch.Tensor):
        update_list: Iterable = [update]
        lang_list: Iterable = [lang_grad]
    else:
        update_list = list(update)
        lang_list = list(lang_grad)
    dot = 0.0
    norm_u = 0.0
    norm_g = 0.0
    for u, g in zip(update_list, lang_list):
        if u is None or g is None:
            continue
        u_f = u.detach().to(torch.float32).flatten()
        g_f = g.detach().to(torch.float32).flatten()
        dot += float(torch.dot(u_f, g_f))
        norm_u += float(torch.dot(u_f, u_f))
        norm_g += float(torch.dot(g_f, g_f))
    if norm_u <= 0 or norm_g <= 0:
        return 0.0
    sin_alpha = min(1.0, abs(dot) / math.sqrt(norm_u * norm_g))
    return float(math.asin(sin_alpha))


def gradient_cosine(g_a: Sequence[torch.Tensor], g_b: Sequence[torch.Tensor]) -> float:
    """Cosine similarity between two per-parameter gradient lists. Used for telemetry."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(g_a, g_b):
        if a is None or b is None:
            continue
        af = a.detach().to(torch.float32).flatten()
        bf = b.detach().to(torch.float32).flatten()
        dot += float(torch.dot(af, bf))
        norm_a += float(torch.dot(af, af))
        norm_b += float(torch.dot(bf, bf))
    denom = math.sqrt(norm_a * norm_b)
    return 0.0 if denom <= 0 else dot / denom
