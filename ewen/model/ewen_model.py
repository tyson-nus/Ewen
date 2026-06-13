"""Ewen pipeline (paper §3.2–§3.4 + Algorithm 1).

Frozen VQ tokenizer → EEG-token embedding → covariate FiLM (Eq. 2) →
hierarchical position encoding (Eq. 4) → Qwen3.5 backbone with LoRA → output head.

Supports two heads:
    * generation       — autoregressive cross-entropy over a target text region.
    * classification   — 2-layer temporal conv head + linear classifier (paper §N).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .covariate import CovariateProjector
from .neural_transformer import NTConfig, NeuralTransformer
from .norm_ema_quantizer import NormEMAVectorQuantizer
from .orthogonal_lora import attach_lora
from .position import HierarchicalPositionEncoder
from .vq_tokenizer import EwenVQTokenizer


def _default_encoder_cfg(n_embd: int = 768, patch_size: int = 200, n_layer: int = 12) -> NTConfig:
    return NTConfig(
        block_size=1024, patch_size=patch_size, num_classes=0, in_chans=1,
        out_chans=16, n_layer=n_layer, n_head=12, n_embd=n_embd, dropout=0.0,
    )


def _default_decoder_cfg(n_embd: int = 768, patch_size: int = 200, embed_dim: int = 128) -> NTConfig:
    return NTConfig(
        block_size=1024, patch_size=patch_size, num_classes=0, in_chans=embed_dim,
        out_chans=16, n_layer=4, n_head=12, n_embd=n_embd, dropout=0.0,
    )


def load_vq_tokenizer(ckpt_path: Optional[str], patch_size: int = 200,
                      n_embed: int = 8192, embed_dim: int = 128) -> EwenVQTokenizer:
    enc_cfg = _default_encoder_cfg(patch_size=patch_size)
    dec_cfg = _default_decoder_cfg(patch_size=patch_size, embed_dim=embed_dim)
    model = EwenVQTokenizer(enc_cfg, dec_cfg, n_embed=n_embed, embed_dim=embed_dim)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("model", state)
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[load_vq_tokenizer] missing keys (ok if expected): {len(missing)}")
        if unexpected:
            print(f"[load_vq_tokenizer] unexpected keys: {len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TemporalConvHead(nn.Module):
    """2- or 4-layer temporal-conv classification head (paper §N)."""

    def __init__(self, d_in: int, num_classes: int, num_layers: int = 2,
                 hidden: int = 128, kernel: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_in
        for _ in range(num_layers):
            layers.append(nn.Conv1d(prev, hidden, kernel_size=kernel, padding=kernel // 2))
            layers.append(nn.GELU())
            prev = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d). Apply temporal convs over time, mean-pool, classify.
        h = self.body(x.transpose(1, 2))  # (B, hidden, T)
        h = h.mean(dim=-1)
        return self.head(h)


@dataclass
class EwenOutput:
    loss: torch.Tensor
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


class EwenModel(nn.Module):
    """End-to-end Ewen forward pass.

    Constructor arguments
    ---------------------
    backbone_name    : HF id (e.g. ``Qwen/Qwen3.5-0.8B``) — passed to ``AutoModelForCausalLM``.
    vq_ckpt          : path to the Stage-1 VQ checkpoint; loaded frozen.
    covariate_dim    : ignored, kept for API symmetry (the projector reads 18 + 4).
    lora_rank        : if > 0, attach a LoRA adapter (Eq. 10) on top of the backbone.
    enable_classifier: if not None, attach a temporal-conv head with this many classes.
    classifier_layers: depth of the temporal-conv head (paper §N: 2 default, 4 for SHU-MI / Mumtaz).
    mask_ratio       : training-time random masking applied to EEG tokens (paper §L: 0.2).
    """

    def __init__(self, backbone_name: str = "Qwen/Qwen3.5-0.8B",
                 vq_ckpt: Optional[str] = None,
                 patch_size: int = 200,
                 covariate_dim: int = 18,
                 lora_rank: int = 0,
                 enable_classifier: Optional[int] = None,
                 classifier_layers: int = 2,
                 mask_ratio: float = 0.2,
                 max_time_patches: int = 64,
                 max_channels: int = 256):
        super().__init__()
        del covariate_dim  # consumed by CovariateProjector defaults
        self.vq = load_vq_tokenizer(vq_ckpt, patch_size=patch_size)
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_name, trust_remote_code=True)
        self.d_hidden = int(self.backbone.config.hidden_size)
        self.mask_ratio = float(mask_ratio)

        # EEG token vocabulary embedding (size K) projected into the backbone hidden size.
        self.eeg_token_embed = nn.Embedding(self.vq.n_embed, self.d_hidden)
        self.eeg_token_embed.weight.data.normal_(mean=0.0, std=0.02)

        self.covariate = CovariateProjector(self.d_hidden)
        self.position = HierarchicalPositionEncoder(
            self.d_hidden, max_time_patches=max_time_patches, max_channels=max_channels,
        )
        # Single learnable prefix-slot for the projected covariate (paper §3.3).
        self.cov_prefix_proj = nn.Linear(self.d_hidden, self.d_hidden)

        if enable_classifier is not None:
            self.classifier_head = TemporalConvHead(
                d_in=self.d_hidden, num_classes=enable_classifier,
                num_layers=classifier_layers,
            )
        else:
            self.classifier_head = None

        if lora_rank > 0:
            self.backbone = attach_lora(self.backbone, r=lora_rank, alpha=2 * lora_rank)

    # ----------------------------------------------------------------- helpers
    @torch.no_grad()
    def tokenize_eeg(self, eeg: torch.Tensor, input_chans: torch.Tensor,
                     input_times: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        return self.vq.get_codebook_indices(eeg, input_chans, input_times, input_mask)

    def _maybe_mask(self, eeg_emb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.mask_ratio <= 0.0:
            return eeg_emb
        keep = (torch.rand_like(valid_mask.float()) > self.mask_ratio) | (~valid_mask)
        return eeg_emb * keep.unsqueeze(-1)

    @property
    def backbone_dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def build_eeg_embeddings(self, eeg: torch.Tensor, input_chans: torch.Tensor,
                              input_times: torch.Tensor, input_mask: torch.Tensor,
                              descriptors: torch.Tensor, modality_mask: torch.Tensor,
                              stage: torch.Tensor):
        codes = self.tokenize_eeg(eeg, input_chans, input_times, input_mask)
        e = self.eeg_token_embed(codes)  # (B, T_z, d)
        c = self.covariate(descriptors, modality_mask)
        e = self.covariate.apply_film(e, c)
        pos = self.position(input_times, input_chans, stage, c)
        h = self._maybe_mask(e + pos, input_mask)
        prefix = self.cov_prefix_proj(c).unsqueeze(1)
        full = torch.cat([prefix, h], dim=1).to(self.backbone_dtype)
        return full, c

    # --------------------------------------------------------------- forward modes
    def forward_classification(self, eeg, input_chans, input_times, input_mask,
                                descriptors, modality_mask, stage, labels=None) -> EwenOutput:
        assert self.classifier_head is not None, "classifier_head not configured"
        eeg_emb, _ = self.build_eeg_embeddings(
            eeg, input_chans, input_times, input_mask, descriptors, modality_mask, stage,
        )
        out = self.backbone(inputs_embeds=eeg_emb, output_hidden_states=True,
                            use_cache=False)
        hidden = out.hidden_states[-1]  # (B, 1 + T_z, d)
        # Skip the prefix slot, then feed through the temporal-conv head in float32.
        logits = self.classifier_head(hidden[:, 1:].float())
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return EwenOutput(
            loss=loss if loss is not None else torch.tensor(0.0, device=logits.device),
            logits=logits, hidden_states=hidden,
        )

    def forward_generation(self, eeg, input_chans, input_times, input_mask,
                            descriptors, modality_mask, stage,
                            prompt_ids: torch.Tensor, target_ids: torch.Tensor,
                            prompt_attention_mask: Optional[torch.Tensor] = None,
                            target_attention_mask: Optional[torch.Tensor] = None) -> EwenOutput:
        eeg_emb, _ = self.build_eeg_embeddings(
            eeg, input_chans, input_times, input_mask, descriptors, modality_mask, stage,
        )
        emb_layer = self.backbone.get_input_embeddings()
        prompt_emb = emb_layer(prompt_ids)
        target_emb = emb_layer(target_ids)
        inputs_embeds = torch.cat([eeg_emb, prompt_emb, target_emb], dim=1)

        B, T_eeg, _ = eeg_emb.shape
        labels = torch.full(
            (B, inputs_embeds.size(1)), -100, dtype=torch.long, device=inputs_embeds.device,
        )
        target_start = T_eeg + prompt_ids.size(1)
        labels[:, target_start:] = target_ids
        if target_attention_mask is not None:
            labels[:, target_start:] = torch.where(
                target_attention_mask.bool(), target_ids, torch.full_like(target_ids, -100),
            )

        out = self.backbone(inputs_embeds=inputs_embeds, labels=labels,
                            use_cache=False)
        return EwenOutput(loss=out.loss, logits=out.logits)

    def forward_text(self, text_ids: torch.Tensor, text_attention_mask: torch.Tensor) -> EwenOutput:
        """Plain language-modelling pass on text-only mini-batches (for orthogonal update)."""
        labels = text_ids.masked_fill(text_attention_mask == 0, -100)
        out = self.backbone(input_ids=text_ids, attention_mask=text_attention_mask,
                            labels=labels, use_cache=False)
        return EwenOutput(loss=out.loss, logits=out.logits)

    # ------------------------------------------------------------- parameter helpers
    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def lora_parameters(self):
        # LoRA params are flagged by ``peft`` with ``requires_grad=True`` after
        # ``attach_lora``; backbone base weights are frozen.
        return [p for n, p in self.backbone.named_parameters() if p.requires_grad]
