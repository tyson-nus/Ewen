"""Descriptorized physiological covariates (paper §3.2, Eq. 1–2).

Takes the 18 physiological scalars plus a 4-bit modality-presence mask, normalises them,
projects to a latent covariate ``ĉ ∈ R^d``, and applies a FiLM-style modulation
``ẽ_t = (1 + Γ(ĉ)) ⊙ e_t + B(ĉ)`` to every EEG token embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.descriptors import DESCRIPTOR_DIM, NUM_MODALITIES


class CovariateProjector(nn.Module):
    """``CovariateProjector(d_hidden)`` — encodes 18 descriptors into a latent covariate."""

    def __init__(self, d_hidden: int, descriptor_dim: int = DESCRIPTOR_DIM,
                 modality_dim: int = NUM_MODALITIES, mlp_dim: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(descriptor_dim + modality_dim)
        self.proj = nn.Sequential(
            nn.Linear(descriptor_dim + modality_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_hidden),
        )
        # FiLM parameters (scale Γ and shift B) per EEG token.
        self.film = nn.Linear(d_hidden, 2 * d_hidden)

    def forward(self, descriptors: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """Return ``ĉ ∈ R^{B × d_hidden}``."""
        x = torch.cat([descriptors, modality_mask], dim=-1)
        x = self.input_norm(x)
        return self.proj(x)

    def apply_film(self, e: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply FiLM (Eq. 2) to a (B, T, d) token stream using window covariate ``c``."""
        gamma, beta = self.film(c).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # (B, 1, d)
        beta = beta.unsqueeze(1)
        return (1.0 + gamma) * e + beta
