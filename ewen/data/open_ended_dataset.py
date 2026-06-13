"""Open-ended joint training dataset (paper §4.4, Tables 4–7).

Wraps :class:`InstructionDataset` outputs with one of the four sub-task prompts and the
corresponding rendered target text. Sub-tasks: ``label``, ``band``, ``relation``,
``summary``. The four sub-tasks are sampled with equal probability.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from .instruction_dataset import DATASET_REGISTRY, DatasetSpec, InstructionDataset
from .prompts import SUB_TASKS, render_target, sample_answer_prefix, sample_prompt


class OpenEndedDataset(Dataset):
    """One sub-task sample per access; sub-task is sampled uniformly at random."""

    def __init__(self, dataset_names: Sequence[str], roots: Sequence[str], split: str,
                 tasks: Sequence[str] = SUB_TASKS):
        if len(dataset_names) != len(roots):
            raise ValueError("dataset_names and roots must have the same length")
        self.tasks = tuple(tasks)
        self._inner: List[InstructionDataset] = []
        self._spec: List[DatasetSpec] = []
        for name, root in zip(dataset_names, roots):
            spec = DATASET_REGISTRY[name]
            self._inner.append(InstructionDataset(spec, root, split))
            self._spec.append(spec)
        self._cum_len = []
        running = 0
        for ds in self._inner:
            running += len(ds)
            self._cum_len.append(running)

    def __len__(self) -> int:
        return self._cum_len[-1]

    def _locate(self, idx: int):
        for ds_idx, end in enumerate(self._cum_len):
            if idx < end:
                local = idx - (self._cum_len[ds_idx - 1] if ds_idx > 0 else 0)
                return ds_idx, local
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        ds_idx, local_idx = self._locate(idx)
        sample = self._inner[ds_idx][local_idx]
        spec = self._spec[ds_idx]
        rng = random.Random(idx)
        task = rng.choice(self.tasks)
        prompt = sample_prompt(task, spec.name, rng)
        label_text = spec.label_names[sample["label"]]
        prefix = sample_answer_prefix(rng)
        target = render_target(
            task, spec.name, label_text,
            sample["descriptors"].numpy(), sample["modality_mask"].numpy(),
            prefix=prefix, rng=rng,
        )
        sample.update({"task": task, "prompt": prompt, "target": target,
                        "dataset_name": spec.name})
        return sample


def open_ended_collate(samples, tokenizer, max_prompt_len: int = 128,
                       max_target_len: int = 128):
    """Pad a batch of open-ended samples into tensors usable by ``EwenModel.forward_generation``."""
    eeg = torch.stack([s["eeg"] for s in samples])
    input_chans = torch.stack([s["input_chans"] for s in samples])
    input_times = torch.stack([s["input_times"] for s in samples])
    input_mask = torch.stack([s["input_mask"] for s in samples])
    descriptors = torch.stack([s["descriptors"] for s in samples])
    modality_mask = torch.stack([s["modality_mask"] for s in samples])
    stage = torch.stack([s["stage"] for s in samples])

    prompts = [s["prompt"] + " " for s in samples]
    targets = [s["target"] + tokenizer.eos_token for s in samples]
    p = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=max_prompt_len)
    t = tokenizer(targets, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=max_target_len, add_special_tokens=False)
    return {
        "eeg": eeg,
        "input_chans": input_chans,
        "input_times": input_times,
        "input_mask": input_mask,
        "descriptors": descriptors,
        "modality_mask": modality_mask,
        "stage": stage,
        "prompt_ids": p["input_ids"],
        "prompt_attention_mask": p["attention_mask"],
        "target_ids": t["input_ids"],
        "target_attention_mask": t["attention_mask"],
        "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
    }
