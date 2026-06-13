"""Text-aligned EEG VQ tokenizer (paper Appendix B).

Encoder → projection (d_q=128) → EMA-updated normalized VQ (K=8192) → decoder.
Reconstruction uses smooth-L1 (Huber) over the raw signal. Contrastive alignment and
domain-adversarial confusion are implemented in the training wrapper.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .neural_transformer import NTConfig, NeuralTransformer
from .norm_ema_quantizer import NormEMAVectorQuantizer


@dataclass
class VQConfig:
    n_embed: int = 8192
    embed_dim: int = 128
    patch_size: int = 200
    encoder: NTConfig = None
    decoder: NTConfig = None


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class EwenVQTokenizer(nn.Module):
    """Encoder + EMA quantizer + Huber-loss reconstruction decoder (paper Eq. 15–20)."""

    def __init__(self, encoder_cfg: NTConfig, decoder_cfg: NTConfig,
                 n_embed: int = 8192, embed_dim: int = 128, decay: float = 0.99):
        super().__init__()
        # Decoder takes codebook vectors as input -> overwrite its in_chans.
        if decoder_cfg.in_chans != embed_dim:
            decoder_cfg.in_chans = embed_dim
        self.encoder = NeuralTransformer(encoder_cfg)
        self.decoder = NeuralTransformer(decoder_cfg)
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0,
            kmeans_init=True, decay=decay,
        )
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_cfg.n_embd, encoder_cfg.n_embd),
            nn.Tanh(),
            nn.Linear(encoder_cfg.n_embd, embed_dim),
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_cfg.n_embd, decoder_cfg.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_cfg.n_embd, decoder_cfg.patch_size),
        )
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self._init_task_layers()

    def _init_task_layers(self) -> None:
        for m in [self.encode_task_layer, self.decode_task_layer]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    # ------------------------------------------------------------- encode / decode
    def encode(self, x: torch.Tensor, input_chans=None, input_times=None, mask=None):
        feat = self.encoder(x, input_chans, input_times, mask, return_all_tokens=True)
        proj = self.encode_task_layer(feat)
        z_q, vq_loss, indices = self.quantize(proj)
        return z_q, indices, vq_loss, feat

    def decode(self, z_q: torch.Tensor, input_chans=None, input_times=None, mask=None) -> torch.Tensor:
        feat = self.decoder(z_q, input_chans, input_times, mask, return_all_tokens=True)
        return self.decode_task_layer(feat)

    @torch.no_grad()
    def get_codebook_indices(self, x: torch.Tensor, input_chans=None, input_times=None,
                              input_mask=None) -> torch.Tensor:
        attn_mask = None if input_mask is None else input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        _, indices, _, _ = self.encode(x, input_chans, input_times, attn_mask)
        return indices.view(x.size(0), x.size(1))

    # --------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, y_raw: torch.Tensor, input_chans=None,
                input_times=None, input_mask=None):
        attn_mask = None if input_mask is None else input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        z_q, _, vq_loss, encoder_feat = self.encode(x, input_chans, input_times, attn_mask)
        recon = self.decode(z_q, input_chans, input_times, attn_mask)
        if input_mask is not None:
            mask = input_mask.unsqueeze(-1).expand_as(recon).float()
            rec_loss = F.smooth_l1_loss(recon * mask, y_raw * mask)
        else:
            rec_loss = F.smooth_l1_loss(recon, y_raw)
        total = vq_loss + rec_loss
        log = {
            "vq_loss": vq_loss.detach().mean(),
            "rec_loss": rec_loss.detach().mean(),
            "total_loss": total.detach().mean(),
        }
        return total, encoder_feat, log

    # ------------------------------------------------------------------ optim utils
    def configure_optimizers(self, weight_decay: float, lr: float,
                              betas: tuple = (0.9, 0.999), device_type: str = "cuda"):
        decay, nodecay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
