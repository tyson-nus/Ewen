from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EncoderConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    patch_size: int = 200
    out_chans: int = 16
    bias: bool = False
    max_channels: int = 256
    max_times: int = 64


class PatchConv(nn.Module):
    def __init__(self, width=768, out_chans=16, patch_size=200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_chans, (1, 15), (1, 8), (0, 7))
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, (1, 3), padding=(0, 1))
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, (1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.l = nn.Sequential(nn.Linear(out_chans * math.ceil(patch_size / 8), width), nn.GELU())

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2), (self.conv3, self.norm3)):
            x = F.gelu(norm(conv(x)))
        x = x.permute(0, 2, 3, 1).flatten(2)
        return self.l(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head = cfg.n_head

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = [z.view(b, n, self.n_head, d // self.n_head).transpose(1, 2)
                   for z in self.c_attn(x).chunk(3, -1)]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :].bool())
        return self.c_proj(y.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))


class GridEncoder(nn.Module):
    def __init__(self, cfg, input_dim=None):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchConv(cfg.n_embd, cfg.out_chans, cfg.patch_size) if input_dim is None else nn.Linear(input_dim, cfg.n_embd)
        self.pos_embed = nn.Embedding(cfg.max_channels, cfg.n_embd)
        self.time_embed = nn.Embedding(cfg.max_times, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.fc_norm = nn.LayerNorm(cfg.n_embd, eps=1e-6)

    def _dense(self, x, chans, times, mask):
        h = self.patch_embed(x) + self.pos_embed(chans) + self.time_embed(times)
        for block in self.blocks:
            h = block(h, mask)
        return self.fc_norm(h) * mask[..., None]

    def forward(self, x, chans, times, mask):


        if not mask.bool().all() and isinstance(self.patch_embed, PatchConv):
            rows = []
            for i in range(x.size(0)):
                keep = mask[i].bool()
                h = self._dense(x[i:i+1, keep], chans[i:i+1, keep], times[i:i+1, keep], torch.ones(1, int(keep.sum()), device=x.device, dtype=torch.bool))
                padded = h.new_zeros(x.size(1), self.cfg.n_embd)
                padded[keep] = h[0]
                rows.append(padded)
            return torch.stack(rows)
        return self._dense(x, chans, times, mask)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenCheckpointTokenizer(nn.Module):

    def __init__(self, checkpoint):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = ckpt.get("model", ckpt)
        state = {}
        for name, tensor in source.items():
            name = name.removeprefix("_orig_mod.").removeprefix("VQ.")
            if name.startswith(("encoder.", "encode_task_layer.")) or name == "quantize.embedding.weight":
                state[name] = tensor
        args = ckpt.get("encoder_args", {})
        cfg = EncoderConfig(**{k: args[k] for k in EncoderConfig.__dataclass_fields__ if k in args})
        if "quantize.embedding.weight" not in state:
            raise ValueError("Checkpoint has no complete EEG codebook")
        self.n_embed, dim = state["quantize.embedding.weight"].shape
        self.encoder = GridEncoder(cfg)
        self.encode_task_layer = nn.Sequential(nn.Linear(cfg.n_embd, cfg.n_embd), nn.Tanh(), nn.Linear(cfg.n_embd, dim))
        self.quantize = nn.Module()
        self.quantize.embedding = nn.Embedding(self.n_embed, dim)

        self.load_state_dict(state, strict=True)
        self.provenance = {"format": "legacy-vq-encoder", "sha256": sha256_file(checkpoint), "encoder": asdict(cfg), "codebook_size": self.n_embed}
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def get_codebook_indices(self, eeg, input_chans, input_times, input_mask):
        feat = self.encoder(eeg, input_chans, input_times, input_mask)
        latent = F.normalize(self.encode_task_layer(feat).float(), dim=-1)
        codes = F.normalize(self.quantize.embedding.weight.float(), dim=-1)
        flat = latent.flatten(0, 1)
        ids = torch.cat([(block @ codes.T).argmax(-1) for block in flat.split(1024)])
        return ids.reshape(eeg.shape[:2]).masked_fill(~input_mask.bool(), 0)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class SphericalEMA(nn.Module):
    def __init__(self, n_embed=8192, dim=128, decay=.99):
        super().__init__()
        self.n_embed, self.dim, self.decay = n_embed, dim, decay
        self.register_buffer("codebook", F.normalize(torch.randn(n_embed, dim), dim=-1))
        self.register_buffer("counts", torch.zeros(n_embed))
        self.register_buffer("sums", self.codebook.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, points):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        if rank == 0:
            generator = torch.Generator(device=points.device).manual_seed(0)
            centers = points[torch.randint(len(points), (self.n_embed,), generator=generator, device=points.device)].clone()
            for _ in range(10):
                ids = torch.cat([(part @ centers.T).argmax(-1) for part in points.split(1024)])
                sums = torch.zeros_like(centers).index_add_(0, ids, points)
                counts = torch.bincount(ids, minlength=self.n_embed)
                centers = torch.where((counts > 0)[:, None], F.normalize(sums, dim=-1), centers)
            self.codebook.copy_(centers)
        if distributed:
            torch.distributed.broadcast(self.codebook, 0)
        self.sums.copy_(self.codebook)
        self.counts.fill_(1)
        self.initialized.fill_(True)

    def forward(self, u, mask):
        valid = F.normalize(u[mask.bool()].float(), dim=-1)
        if not len(valid):
            raise ValueError("VQ requires valid EEG tokens")
        if self.training and not self.initialized:
            self.initialize(valid.detach())
        ids = torch.cat([(part @ self.codebook.T).argmax(-1) for part in valid.split(1024)])
        chosen = self.codebook[ids]
        loss = F.mse_loss(valid, chosen.detach()) + F.mse_loss(valid.detach(), chosen)
        if self.training:
            with torch.no_grad():
                sums = torch.zeros_like(self.sums).index_add_(0, ids, valid.detach())
                counts = torch.bincount(ids, minlength=self.n_embed).float()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(sums)
                    torch.distributed.all_reduce(counts)
                self.sums.mul_(self.decay).add_(sums, alpha=1-self.decay)
                self.counts.mul_(self.decay).add_(counts, alpha=1-self.decay)
                self.codebook.copy_(F.normalize(self.sums / self.counts.clamp_min(1e-5)[:, None], dim=-1))
        q = u.new_zeros(u.shape)
        q[mask.bool()] = valid.to(u.dtype) + (chosen.to(u.dtype) - valid.to(u.dtype)).detach()
        all_ids = torch.zeros(mask.shape, dtype=torch.long, device=u.device)
        all_ids[mask.bool()] = ids
        return q, all_ids, loss


class PaperVQTokenizer(nn.Module):
    def __init__(self, cfg=None, n_embed=8192, code_dim=128, text_dim=384):
        super().__init__()
        cfg = cfg or EncoderConfig()
        self.cfg, self.n_embed, self.text_dim = cfg, n_embed, text_dim
        self.encoder = GridEncoder(cfg)
        self.encode_task_layer = nn.Sequential(nn.Linear(cfg.n_embd, cfg.n_embd), nn.Tanh(), nn.Linear(cfg.n_embd, code_dim))
        decoder_cfg = EncoderConfig(**{**asdict(cfg), "n_layer": 4})
        self.decoder = GridEncoder(decoder_cfg, code_dim)
        self.raw_head = nn.Linear(cfg.n_embd, cfg.patch_size)
        self.quantizer = SphericalEMA(n_embed, code_dim)
        self.eeg_projection = nn.Linear(cfg.n_embd, 512)
        self.text_projection = nn.Linear(text_dim, 512)
        self.text_to_encoder = nn.Linear(text_dim, cfg.n_embd)
        self.domain_classifier = nn.Sequential(nn.Linear(cfg.n_embd, 256), nn.GELU(), nn.Linear(256, 2))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1/.07)))

    def forward(self, eeg, input_chans, input_times, input_mask, text_features, text_tokens=None, progress=0.):
        h = self.encoder(eeg, input_chans, input_times, input_mask)
        u = self.encode_task_layer(h)
        q, ids, vq = self.quantizer(u, input_mask)
        reconstruction = self.raw_head(self.decoder(q, input_chans, input_times, input_mask))
        rec = F.smooth_l1_loss(reconstruction[input_mask.bool()], eeg[input_mask.bool()])
        pooled = (h * input_mask[..., None]).sum(1) / input_mask.sum(1, keepdim=True).clamp_min(1)
        eeg_f = F.normalize(self.eeg_projection(pooled).float(), dim=-1)
        text_f = F.normalize(self.text_projection(text_features.detach()).float(), dim=-1)
        if len(eeg) < 2:
            raise ValueError("SigLIP alignment requires at least two paired windows")
        logits = self.log_temperature.exp().clamp(max=100) * eeg_f @ text_f.T
        targets = torch.eye(len(eeg), device=eeg.device)
        weight = torch.tensor(float(len(eeg)-1), device=eeg.device)
        contrastive = .5 * (F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight) + F.binary_cross_entropy_with_logits(logits.T, targets, pos_weight=weight))
        text_states = text_features if text_tokens is None else text_tokens.flatten(0, 1)
        text_h = self.text_to_encoder(text_states.detach())
        eeg_h = h[input_mask.bool()]

        n = min(len(eeg_h), len(text_h))
        if n < 1:
            raise ValueError("Domain alignment has no text tokens")
        eeg_idx = torch.randperm(len(eeg_h), device=eeg.device)[:n]
        txt_idx = torch.randperm(len(text_h), device=eeg.device)[:n]
        domain = torch.cat((eeg_h[eeg_idx], text_h[txt_idx]))
        labels = torch.cat((torch.ones(n, device=eeg.device, dtype=torch.long), torch.zeros(n, device=eeg.device, dtype=torch.long)))
        alpha = 2/(1+math.exp(-10*progress))-1
        dom = F.cross_entropy(self.domain_classifier(Reverse.apply(domain, alpha)), labels)
        loss = rec + vq + contrastive + .5 * dom
        return {"loss": loss, "rec": rec, "vq": vq, "contrastive": contrastive, "domain": dom, "codes": ids}

    @torch.no_grad()
    def get_codebook_indices(self, eeg, input_chans, input_times, input_mask):
        h = self.encoder(eeg, input_chans, input_times, input_mask)

        if not self.quantizer.initialized:
            raise ValueError("Tokenizer codebook is untrained")
        was_training = self.quantizer.training
        self.quantizer.eval()
        _, ids, _ = self.quantizer(self.encode_task_layer(h), input_mask)
        self.quantizer.train(was_training)
        return ids


def load_tokenizer(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") == "ewen-paper-vq-v1":
        model = PaperVQTokenizer(EncoderConfig(**checkpoint["encoder_config"]), checkpoint["n_embed"], checkpoint["code_dim"], checkpoint["text_dim"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.provenance = {"format": checkpoint["format"], "sha256": sha256_file(path), "training_manifest": checkpoint.get("training_manifest", {})}
        return model.requires_grad_(False).eval()
    return FrozenCheckpointTokenizer(path)
