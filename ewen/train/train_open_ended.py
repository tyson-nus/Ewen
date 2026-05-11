"""Stage 4 — open-ended joint training with orthogonal update adaptation (paper §3.4).

Trains a LoRA-adapted Qwen3.5 backbone on the three open-ended datasets (Mumtaz, Mental
Arithmetic, SHU-MI) and projects each EEG-task gradient orthogonal to a
language-reference gradient sampled from OpenWebText.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from ..data.open_ended_dataset import OpenEndedDataset, open_ended_collate
from ..model.ewen_model import EwenModel
from ..model.orthogonal_lora import (
    collect_grads,
    project_gradients_orthogonal,
    trainable_lora_params,
)
from ..utils.metrics import leakage_angle
from ..utils.seeding import seed_everything
from ..utils.tokenizer import load_qwen_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: open-ended training with orthogonal LoRA.")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Subset of {Mumtaz, Mental-Arithmetic, SHU-MI, TUAB, ...}.")
    parser.add_argument("--roots", nargs="+", required=True,
                        help="Per-dataset root directory; must match --datasets.")
    parser.add_argument("--vq_ckpt", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--ortho_text", type=str, default=None,
                        help="Path to a plain-text file used as the OpenWebText language probe.")
    parser.add_argument("--ortho_text_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_prompt_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=128)
    return parser.parse_args()


class _LanguageProbe:
    """Cyclic text-only mini-batch source for the orthogonal update."""

    def __init__(self, tokenizer, text_path: Optional[str], seq_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.use_random = text_path is None or not os.path.exists(text_path)
        if not self.use_random:
            with open(text_path, "r", encoding="utf-8") as handle:
                self.corpus = handle.read()
            self.ids = self.tokenizer(self.corpus, return_tensors="pt",
                                       add_special_tokens=False)["input_ids"].squeeze(0)
            if self.ids.numel() < self.seq_len:
                self.use_random = True

    def sample(self, device):
        if self.use_random:
            vocab = self.tokenizer.vocab_size
            ids = torch.randint(0, vocab, (self.batch_size, self.seq_len), device=device)
            mask = torch.ones_like(ids)
            return ids, mask
        n = self.ids.numel() - self.seq_len
        starts = torch.randint(0, n, (self.batch_size,))
        ids = torch.stack([self.ids[s: s + self.seq_len] for s in starts]).to(device)
        return ids, torch.ones_like(ids)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if len(args.datasets) != len(args.roots):
        raise ValueError("--datasets and --roots must have the same length")

    tokenizer = load_qwen_tokenizer(args.backbone)
    dataset = OpenEndedDataset(args.datasets, args.roots, split="train")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=lambda s: open_ended_collate(
            s, tokenizer, max_prompt_len=args.max_prompt_len,
            max_target_len=args.max_target_len,
        ),
    )
    print(f"[open_ended] dataset: {len(dataset)} / batches: {len(loader)}")

    model = EwenModel(
        backbone_name=args.backbone, vq_ckpt=args.vq_ckpt,
        lora_rank=args.lora_rank, enable_classifier=None, mask_ratio=args.mask_ratio,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.1,
    )

    probe = _LanguageProbe(tokenizer, args.ortho_text, args.ortho_text_seq_len, args.batch_size)
    lora_params = trainable_lora_params(model.backbone)

    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            # ---- EEG-task gradient ----
            out_task = model.forward_generation(
                batch["eeg"], batch["input_chans"], batch["input_times"],
                batch["input_mask"], batch["descriptors"], batch["modality_mask"],
                batch["stage"], batch["prompt_ids"], batch["target_ids"],
                target_attention_mask=batch["target_attention_mask"],
            )
            optimizer.zero_grad(set_to_none=True)
            out_task.loss.backward(retain_graph=False)
            g_task = collect_grads(lora_params)

            # ---- Language-reference gradient ----
            text_ids, text_mask = probe.sample(device)
            out_lang = model.forward_text(text_ids, text_mask)
            optimizer.zero_grad(set_to_none=True)
            out_lang.loss.backward()
            g_lang = collect_grads(lora_params)

            # Leakage angle before vs. after projection — paper Eq. 54 / Figure 3.
            angle_before = leakage_angle(g_task, g_lang)
            cos = project_gradients_orthogonal(lora_params, g_task, g_lang)
            projected_grads = [p.grad for p in lora_params]
            angle_after = leakage_angle(projected_grads, g_lang)

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % 5 == 0 or args.max_steps > 0:
                deg_before = angle_before * 180.0 / 3.141592653589793
                deg_after = angle_after * 180.0 / 3.141592653589793
                print(f"[open] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss_gen={float(out_task.loss):.4f} "
                      f"loss_lang={float(out_lang.loss):.4f} "
                      f"cos(g_task,g_lang)={cos:+.3f} "
                      f"leak_deg={deg_before:.2f}->{deg_after:.2f}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    torch.save({
        "model_state": {k: v for k, v in model.state_dict().items()
                        if "backbone.base_model" not in k},
        "lora_state": {k: v for k, v in model.state_dict().items()
                        if "lora_" in k},
        "args": vars(args),
    }, ckpt_path)
    print(f"[train_open_ended] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
