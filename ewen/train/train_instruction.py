"""Stage 3 — closed-vocabulary instruction tuning (paper §N).

One dataset at a time. Trains an EwenModel with a 2-layer (or 4-layer for SHU-MI /
Mumtaz) temporal-conv classification head, lr 2e-5, EEG input mask ratio 0.2.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from ..data.instruction_dataset import DATASET_REGISTRY, InstructionDataset
from ..model.ewen_model import EwenModel
from ..utils.metrics import binary_metrics, multiclass_metrics
from ..utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: closed-vocabulary instruction tuning.")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--root", required=True)
    parser.add_argument("--vq_ckpt", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--classifier_layers", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_eval_steps", type=int, default=-1,
                        help="Cap validation batches (useful for smoke runs).")
    return parser.parse_args()


def _move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _collate(samples):
    keys = ("eeg", "input_chans", "input_times", "input_mask", "descriptors",
            "modality_mask", "stage")
    out = {k: torch.stack([s[k] for s in samples]) for k in keys}
    out["label"] = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    out["key"] = [s["key"] for s in samples]
    return out


def _evaluate(model: EwenModel, loader: DataLoader, device, is_binary: bool,
              max_steps: int = -1):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            out = model.forward_classification(
                batch["eeg"], batch["input_chans"], batch["input_times"],
                batch["input_mask"], batch["descriptors"], batch["modality_mask"],
                batch["stage"], labels=batch["label"],
            )
            probs = torch.softmax(out.logits, dim=-1)
            if is_binary:
                scores.append(probs[:, 1].detach().cpu().numpy())
            else:
                scores.append(probs.detach().cpu().numpy())
            targets.append(batch["label"].detach().cpu().numpy())
            if max_steps > 0 and step + 1 >= max_steps:
                break
    model.train()
    y_score = np.concatenate(scores)
    y_true = np.concatenate(targets)
    return (binary_metrics(y_true, y_score) if is_binary
            else multiclass_metrics(y_true, y_score))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    spec = DATASET_REGISTRY[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = InstructionDataset(spec, args.root, split="train")
    try:
        val_ds = InstructionDataset(spec, args.root, split="val")
    except (FileNotFoundError, RuntimeError):
        val_ds = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True,
                   collate_fn=_collate)
        if val_ds is not None else None
    )

    model = EwenModel(
        backbone_name=args.backbone, vq_ckpt=args.vq_ckpt,
        patch_size=spec.patch_size, lora_rank=0,
        enable_classifier=spec.num_classes, classifier_layers=args.classifier_layers,
        mask_ratio=args.mask_ratio,
    ).to(device)
    # Backbone frozen; only EEG-side + head are trained.
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.1,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = _move_batch(batch, device)
            out = model.forward_classification(
                batch["eeg"], batch["input_chans"], batch["input_times"],
                batch["input_mask"], batch["descriptors"], batch["modality_mask"],
                batch["stage"], labels=batch["label"],
            )
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
                print(f"[instr {args.dataset}] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss={float(loss):.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if val_loader is not None:
        eval_cap = args.max_eval_steps if args.max_eval_steps > 0 else (
            2 if args.max_steps > 0 else -1
        )
        metrics = _evaluate(model, val_loader, device, spec.is_binary,
                            max_steps=eval_cap)
        print(f"[instr {args.dataset}] val metrics: {metrics}")

    ckpt_path = os.path.join(args.ckpt_dir, f"{args.dataset}_ckpt.pt")
    torch.save({
        "model_state": {k: v for k, v in model.state_dict().items() if "backbone" not in k},
        "args": vars(args),
    }, ckpt_path)
    print(f"[train_instruction] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
