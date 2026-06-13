"""Normalized EMA vector quantizer used by the Ewen VQ tokenizer (paper Appendix B)."""

from __future__ import annotations

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10, use_cosine_sim: bool = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
            dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens: int, codebook_dim: int, decay: float = 0.99, eps: float = 1e-5, kmeans_init: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if not kmeans_init:
            weight = l2norm(torch.randn(num_tokens, codebook_dim))
        else:
            weight = torch.zeros(num_tokens, codebook_dim)
        self.register_buffer("initted", torch.tensor([not kmeans_init], dtype=torch.float32))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data: torch.Tensor) -> None:
        if bool(self.initted.item()):
            return
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.tensor([1.0]))

    def forward(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.weight)


def norm_ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):
    """EMA-updated normalized VQ. ``K=8192, dim=128, decay=0.99`` per Appendix B."""

    def __init__(self, n_embed: int, embedding_dim: int, beta: float = 1.0,
                 decay: float = 0.99, eps: float = 1e-5,
                 statistic_code_usage: bool = True, kmeans_init: bool = True):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(n_embed, embedding_dim, decay, eps, kmeans_init)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer("cluster_size", torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device) -> None:
        if self.statistic_code_usage:
            self.register_buffer("cluster_size", torch.zeros(self.num_tokens, device=device))

    def forward(self, z: torch.Tensor):
        z = l2norm(z)
        z_flat = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flat)
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flat, self.embedding.weight)
        )
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = z_flat.t() @ encodings
            self.all_reduce_fn(embed_sum)
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices
