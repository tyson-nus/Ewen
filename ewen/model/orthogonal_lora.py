"""Orthogonal-update LoRA adapter (paper §3.4, Eq. 10–12).

A thin wrapper around the ``peft`` LoRA implementation plus a helper that projects the
EEG-task gradient orthogonal to a language-reference gradient on a per-parameter basis:

    g⊥ = g_task − ⟨g_task, g_lang⟩ / (‖g_lang‖² + ε) · g_lang
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ewen requires the `peft` package for orthogonal-update LoRA adaptation."
    ) from exc

DEFAULT_TARGET_MODULES: Sequence[str] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def attach_lora(model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.0,
                target_modules: Sequence[str] = DEFAULT_TARGET_MODULES) -> nn.Module:
    """Wrap ``model`` (a HF causal-LM backbone) in a LoRA adapter."""
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    return get_peft_model(model, cfg)


def trainable_lora_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def collect_grads(params: Iterable[nn.Parameter]) -> list[torch.Tensor | None]:
    return [p.grad.detach().clone() if p.grad is not None else None for p in params]


def project_gradients_orthogonal(params: Sequence[nn.Parameter],
                                  task_grads: Sequence[torch.Tensor | None],
                                  lang_grads: Sequence[torch.Tensor | None],
                                  eps: float = 1e-8) -> float:
    """Write back ``g⊥`` into ``p.grad`` for each parameter; return the cosine of the
    angle between the aggregate task and language gradients (for telemetry)."""
    if len(params) != len(task_grads) or len(params) != len(lang_grads):
        raise ValueError("params / task_grads / lang_grads must all be the same length")
    dot = 0.0
    norm_lang = 0.0
    norm_task = 0.0
    for g_task, g_lang in zip(task_grads, lang_grads):
        if g_task is None or g_lang is None:
            continue
        dot += float(torch.sum(g_task * g_lang))
        norm_lang += float(torch.sum(g_lang * g_lang))
        norm_task += float(torch.sum(g_task * g_task))
    coeff = dot / (norm_lang + eps)
    for p, g_task, g_lang in zip(params, task_grads, lang_grads):
        if g_task is None:
            p.grad = None
            continue
        if g_lang is None:
            p.grad = g_task
            continue
        p.grad = g_task - coeff * g_lang
    denom = (norm_task ** 0.5) * (norm_lang ** 0.5) + eps
    return dot / denom
