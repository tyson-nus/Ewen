"""Stage 2 — Ewen pretraining (paper Appendix N).

Joint EEG-token + paired-text next-token prediction on a frozen Qwen3.5 backbone. The
EEG side is mapped through the FiLM/PE pipeline; the text side is supervised directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.pretrain_dataset import PretrainEEGDataset
from ..model.ewen_model import EwenModel
from ..utils.seeding import seed_everything
from ..utils.tokenizer import load_qwen_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Ewen pretraining.")
    parser.add_argument("--vq_ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_json", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_text_len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument("--eeg_T", type=int, default=2000)
    parser.add_argument("--eeg_C", type=int, default=16)
    return parser.parse_args()


def collate_pretrain(samples, tokenizer, max_text_len: int):
    keys = ("eeg", "input_chans", "input_times", "input_mask", "descriptors",
            "modality_mask", "stage")
    batch = {k: torch.stack([s[k] for s in samples]) for k in keys}
    texts = [s["text"] + tokenizer.eos_token for s in samples]
    tok = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                    max_length=max_text_len, add_special_tokens=False)
    batch["text_ids"] = tok["input_ids"]
    batch["text_mask"] = tok["attention_mask"]
    return batch


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = load_qwen_tokenizer(args.backbone)
    dataset = PretrainEEGDataset(
        data_root=args.data_root, meta_json=args.meta_json,
        split=args.train_split, expected_T=args.eeg_T, expected_C=args.eeg_C,
        patch_size=args.patch_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda s: collate_pretrain(s, tokenizer, args.max_text_len),
    )
    print(f"[train_pretrain] dataset: {len(dataset)} / batches: {len(loader)}")

    model = EwenModel(
        backbone_name=args.backbone, vq_ckpt=args.vq_ckpt,
        patch_size=args.patch_size, lora_rank=0, enable_classifier=None,
    ).to(device)
    # Backbone is frozen during pretraining (paper Appendix N) – only EEG-side modules
    # and the EEG embedding table receive gradient updates.
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.1,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Build EEG embeddings + cross-modal mixture: [EEG, <eos>, text]
            eeg_emb, _ = model.build_eeg_embeddings(
                batch["eeg"], batch["input_chans"], batch["input_times"],
                batch["input_mask"], batch["descriptors"], batch["modality_mask"],
                batch["stage"],
            )
            text_emb = model.backbone.get_input_embeddings()(batch["text_ids"])
            inputs_embeds = torch.cat([eeg_emb, text_emb], dim=1)
            labels = torch.full(
                (inputs_embeds.size(0), inputs_embeds.size(1)), -100,
                dtype=torch.long, device=device,
            )
            text_start = eeg_emb.size(1)
            labels[:, text_start:] = torch.where(
                batch["text_mask"].bool(),
                batch["text_ids"],
                torch.full_like(batch["text_ids"], -100),
            )
            out = model.backbone(inputs_embeds=inputs_embeds, labels=labels, use_cache=False)
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % 10 == 0 or args.max_steps > 0:
                print(f"[pretrain] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss={float(loss):.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    torch.save({
        "model_state": {k: v for k, v in model.state_dict().items() if "backbone" not in k},
        "args": vars(args),
    }, ckpt_path)
    print(f"[train_pretrain] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
