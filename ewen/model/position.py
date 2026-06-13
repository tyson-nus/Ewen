"""Covariate-aware hierarchical position encoding (paper §3.3, Eq. 4).

``h(0)_t = ẽ_t + p_loc_t + p_glob_{w(t)}`` where
    p_loc_t  = PE(τ_t) + IE(u_t)                       — local time + channel id
    p_glob_w = Ψ([ρ(s_w); ĉ_w])                        — harmonic stage map + covariate
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HierarchicalPositionEncoder(nn.Module):
    def __init__(self, d_hidden: int, max_time_patches: int = 64, max_channels: int = 256,
                 num_harmonics: int = 4):
        super().__init__()
        self.local_time = nn.Embedding(max_time_patches, d_hidden)
        self.channel_id = nn.Embedding(max_channels, d_hidden)
        self.num_harmonics = int(num_harmonics)
        # Harmonic stage map ρ(s) ∈ R^{2K} concatenated with the covariate ĉ ∈ R^d.
        self.global_proj = nn.Sequential(
            nn.Linear(2 * num_harmonics + d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def harmonic_stage_map(self, stage: torch.Tensor) -> torch.Tensor:
        """``ρ(s) = [sin(2πks), cos(2πks)]_{k=1..K}`` for stage ∈ [0, 1]."""
        k = torch.arange(1, self.num_harmonics + 1, device=stage.device, dtype=stage.dtype)
        ang = 2 * math.pi * stage.unsqueeze(-1) * k  # (..., K)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, time_ids: torch.Tensor, chan_ids: torch.Tensor,
                stage: torch.Tensor, covariate: torch.Tensor) -> torch.Tensor:
        """time_ids/chan_ids: (B, T_z); stage/covariate: (B,) and (B, d). Returns (B, T_z, d)."""
        p_local = self.local_time(time_ids) + self.channel_id(chan_ids)
        rho = self.harmonic_stage_map(stage)
        global_inp = torch.cat([rho, covariate], dim=-1)
        p_global = self.global_proj(global_inp).unsqueeze(1)  # broadcast over T_z
        return p_local + p_global
