"""Contrastive losses used by the text-aligned VQ tokenizer (paper Eq. 22–24)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _resolve_scale(logit_scale) -> torch.Tensor | float:
    if logit_scale is None:
        return 1.0 / 0.07
    if isinstance(logit_scale, torch.Tensor) and logit_scale.numel() == 1:
        return torch.clamp(logit_scale.exp(), max=100.0)
    return float(logit_scale)


def clip_loss(eeg_embeds: torch.Tensor, text_embeds: torch.Tensor,
              logit_scale=None) -> Tuple[torch.Tensor, dict]:
    """Symmetric InfoNCE CLIP loss (paper Eq. 24)."""
    assert eeg_embeds.shape == text_embeds.shape
    eeg = F.normalize(eeg_embeds, dim=-1)
    txt = F.normalize(text_embeds, dim=-1)
    scale = _resolve_scale(logit_scale)
    logits = scale * eeg @ txt.t()
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    return loss, {"logits": logits.detach()}


def siglip_loss(eeg_embeds: torch.Tensor, text_embeds: torch.Tensor,
                logit_scale=None) -> Tuple[torch.Tensor, dict]:
    """Symmetric SigLIP loss (paper Eq. 22–23): per-pair sigmoid BCE."""
    assert eeg_embeds.shape == text_embeds.shape
    eeg = F.normalize(eeg_embeds, dim=-1)
    txt = F.normalize(text_embeds, dim=-1)
    scale = _resolve_scale(logit_scale)
    logits = scale * eeg @ txt.t()
    B = logits.size(0)
    targets = torch.eye(B, device=logits.device)
    # Pos-weight balances the single positive vs. (B-1) negatives per row.
    pos_weight = torch.full((B,), float(max(1, B - 1)), device=logits.device)
    loss_row = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    loss_col = F.binary_cross_entropy_with_logits(logits.t(), targets, pos_weight=pos_weight)
    return 0.5 * (loss_row + loss_col), {"logits": logits.detach()}
