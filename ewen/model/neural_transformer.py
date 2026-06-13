"""ViT-style EEG encoder used by the text-aligned VQ tokenizer (paper Appendix B)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from .transformer_blocks import Block


class TemporalConv(nn.Module):
    """Patchify a multi-channel EEG window along the temporal axis (patch_size=200)."""

    def __init__(self, in_chans: int = 1, out_chans: int = 8, n_embd: int = 768):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
        # Per-patch feature dim after the three strided convs is 400 (T_patch / 8 * out_chans)
        # for the canonical patch_size=200 case used in the paper.
        self.proj = nn.Sequential(nn.Linear(400, n_embd), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, patch_size) where num_patches == C * (T // patch_size)
        x = x.unsqueeze(1)  # (B, 1, num_patches, patch_size)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, "b c n t -> b n (t c)")
        return self.proj(x)


@dataclass
class NTConfig:
    block_size: int = 1024
    patch_size: int = 200
    num_classes: int = 0
    in_chans: int = 1
    out_chans: int = 16
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class NeuralTransformer(nn.Module):
    """ViT-style encoder/decoder for EEG patches.

    When ``config.in_chans == 1`` the input is patchified by :class:`TemporalConv`; otherwise
    a linear projection is applied — this dual-mode lets us reuse the same class for the
    encoder (raw EEG in) and the decoder (codebook embedding in).
    """

    def __init__(self, config: NTConfig):
        super().__init__()
        self.num_classes = config.num_classes
        if config.in_chans == 1:
            self.patch_embed = TemporalConv(out_chans=config.out_chans, n_embd=config.n_embd)
        else:
            self.patch_embed = nn.Linear(config.in_chans, config.n_embd)
        self.patch_size = config.patch_size

        self.pos_embed = nn.Embedding(256, config.n_embd)
        self.time_embed = nn.Embedding(64, config.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.n_embd, eps=1e-6)
        self.fc_norm = nn.LayerNorm(config.n_embd, eps=1e-6) if config.use_mean_pooling else None
        self.head = nn.Linear(config.n_embd, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.pos_drop = nn.Dropout(p=config.dropout)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.c_proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.c_proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False):
        x = self.patch_embed(x)
        if input_chans is not None:
            x = x + self.pos_embed(input_chans)
        if input_times is not None:
            x = x + self.time_embed(input_times)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x) if return_all_tokens else self.fc_norm(x.mean(1))
        return x

    def forward(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False):
        x = self.forward_features(x, input_chans, input_times, mask, return_all_tokens=return_all_tokens)
        return self.head(x)
