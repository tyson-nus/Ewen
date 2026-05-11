"""Stage 1 — text-aligned EEG VQ tokenizer training (paper Appendix B).

Loss: ``L_rec + L_VQ + λ_c L_contr + λ_d L_dom`` with λ_c = 1.0, λ_d = 0.5.

Example mini-run:
    python -m ewen.train.train_vq \
        --data_root /data2/majy/data/TUAB \
        --meta_json /data2/tianyu/data/tuab_summarytext.json \
        --train_split train --batch_size 2 --epochs 1 --max_steps 2 \
        --lm_name bert-base-uncased --ckpt_dir /tmp/ewen_vq
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from ..data.eeg_text_dataset import EEGTextDataset
from ..model.neural_transformer import NTConfig
from ..model.vq_tokenizer import EwenVQTokenizer, gradient_reverse
from ..utils.losses import siglip_loss
from ..utils.seeding import seed_everything


class MeanPooler(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)
        s = (hidden * mask).sum(dim=1)
        n = mask.sum(dim=1).clamp_min(1e-6)
        return s / n


class TextAlignedVQWrapper(nn.Module):
    """Wraps the VQ tokenizer with a frozen text encoder, SigLIP contrastive head and a
    gradient-reversal domain classifier (paper Eq. 22–25)."""

    def __init__(self, vq: EwenVQTokenizer, text_model_name: str, proj_dim: int = 512):
        super().__init__()
        self.vq = vq
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder.eval()

        self.pooler = MeanPooler()
        eeg_dim = vq.encoder.blocks[0].ln_1.weight.numel()  # n_embd of the encoder
        text_dim = self.text_encoder.config.hidden_size
        self.eeg_proj = nn.Linear(eeg_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        # Domain-confusion classifier (EEG vs. text tokens).
        self.domain_classifier = nn.Sequential(
            nn.Linear(eeg_dim, 256), nn.GELU(), nn.Linear(256, 2),
        )
        # We embed text tokens into the same dimension via the encoder's wte.
        self.text_to_eeg_dim = nn.Linear(text_dim, eeg_dim, bias=False)

    @torch.no_grad()
    def encode_text(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=text_ids, attention_mask=text_mask)
        return self.pooler(out.last_hidden_state, text_mask)

    def forward(self, X, Y_raw, input_chans, input_times, input_mask,
                text_ids, text_mask, alpha: float = 1.0):
        loss_main, encoder_feat, log = self.vq(X, Y_raw, input_chans, input_times, input_mask)
        # SigLIP contrastive (paper Eq. 22–23)
        eeg_pool = self.pooler(encoder_feat, input_mask)
        text_pool = self.encode_text(text_ids, text_mask)
        loss_contr, _ = siglip_loss(
            self.eeg_proj(eeg_pool), self.text_proj(text_pool), self.logit_scale,
        )
        # Domain-adversarial confusion (paper Eq. 25)
        eeg_rev = gradient_reverse(encoder_feat, alpha)
        eeg_logits = self.domain_classifier(eeg_rev)
        eeg_targets = torch.zeros(eeg_logits.shape[:-1], dtype=torch.long, device=eeg_logits.device)
        with torch.no_grad():
            text_emb_raw = self.text_encoder.embeddings.word_embeddings(text_ids)
        text_emb = self.text_to_eeg_dim(text_emb_raw)
        text_emb_rev = gradient_reverse(text_emb, alpha)
        text_logits = self.domain_classifier(text_emb_rev)
        text_targets = torch.ones(text_logits.shape[:-1], dtype=torch.long, device=text_logits.device)
        loss_dom = (
            F.cross_entropy(eeg_logits.reshape(-1, 2), eeg_targets.reshape(-1))
            + F.cross_entropy(text_logits.reshape(-1, 2), text_targets.reshape(-1))
        ) * 0.5
        log["contr_loss"] = loss_contr.detach()
        log["dom_loss"] = loss_dom.detach()
        return loss_main, loss_contr, loss_dom, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: text-aligned EEG VQ tokenizer.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_json", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--lm_name", type=str, default="bert-base-uncased")
    parser.add_argument("--eeg_T", type=int, default=2000)
    parser.add_argument("--eeg_C", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_embed", type=int, default=8192)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_text_len", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--max_steps", type=int, default=-1, help="Cap optimizer steps (smoke runs).")
    parser.add_argument("--lambda_contr", type=float, default=1.0)
    parser.add_argument("--lambda_dom", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    dataset = EEGTextDataset(
        data_root=args.data_root, meta_json=args.meta_json, split=args.train_split,
        lm_name=args.lm_name, max_len=args.max_text_len,
        expected_T=args.eeg_T, expected_C=args.eeg_C,
        patch_size=args.patch_size, block_size=args.block_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"[train_vq] dataset size: {len(dataset)}; batches per epoch: {len(loader)}")

    enc_cfg = NTConfig(block_size=args.block_size, patch_size=args.patch_size,
                       num_classes=0, in_chans=1, out_chans=16,
                       n_layer=12, n_head=12, n_embd=768, dropout=0.0)
    dec_cfg = NTConfig(block_size=args.block_size, patch_size=args.patch_size,
                       num_classes=0, in_chans=args.embed_dim, out_chans=16,
                       n_layer=4, n_head=12, n_embd=768, dropout=0.0)
    vq = EwenVQTokenizer(enc_cfg, dec_cfg, n_embed=args.n_embed, embed_dim=args.embed_dim)
    model = TextAlignedVQWrapper(vq, text_model_name=args.lm_name)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = vq.configure_optimizers(args.weight_decay, args.learning_rate,
                                        betas=(0.9, 0.999),
                                        device_type=device.type)
    total_iters = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, len(loader)), T_mult=1, eta_min=args.learning_rate * 0.1,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            X, Y_raw, chans, times, mask, text_ids, text_mask = batch
            X = X.to(device, non_blocking=True).float()
            Y_raw = Y_raw.to(device, non_blocking=True).float()
            chans = chans.to(device, non_blocking=True)
            times = times.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            text_ids = text_ids.to(device, non_blocking=True)
            text_mask = text_mask.to(device, non_blocking=True)

            progress = global_step / max(1, total_iters)
            alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0  # paper §B.5

            loss_main, loss_contr, loss_dom, log = model(
                X, Y_raw, chans, times, mask, text_ids, text_mask, alpha=alpha,
            )
            loss = loss_main + args.lambda_contr * loss_contr + args.lambda_dom * loss_dom

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 10 == 0 or args.max_steps > 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[vq] epoch {epoch} step {global_step}/{total_iters} "
                      f"loss={float(loss):.4f} (rec={float(log['rec_loss']):.4f}, "
                      f"vq={float(log['vq_loss']):.4f}, contr={float(log['contr_loss']):.4f}, "
                      f"dom={float(log['dom_loss']):.4f}) lr={lr:.2e}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    torch.save({"model": vq.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"[train_vq] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
