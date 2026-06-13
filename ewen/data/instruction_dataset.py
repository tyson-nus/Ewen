"""Closed-vocabulary instruction-tuning dataset (paper Appendix N).

Each entry yields one EEG window with its integer class label and the same descriptor
covariate pathway as :class:`PretrainEEGDataset`. Per-dataset specifics (number of
classes, label-encoder, split layout) are described in :data:`DATASET_REGISTRY`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from .channels import PAD_INDEX, get_channel_indices
from .descriptors import compute_descriptors
from .eeg_text_dataset import DEFAULT_TUAB_CHANNELS


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    is_binary: bool
    splits: Dict[str, str]  # logical -> on-disk subdir
    expected_T: int = 2000
    expected_C: int = 16
    patch_size: int = 200
    block_size: int = 1024
    fs: int = 200
    label_names: Sequence[str] = ("class_0", "class_1")


# Minimal spec covering the eight benchmarks named in paper §4.1 / Appendix D.
# Only ``TUAB`` is wired up against on-disk paths in this release; the rest are present
# in the registry so the CLI can be invoked uniformly but require user-supplied roots.
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "TUAB": DatasetSpec(
        name="TUAB", num_classes=2, is_binary=True,
        splits={"train": "train", "val": "val", "test": "test"},
        label_names=("normal", "abnormal"),
    ),
    "TUEV": DatasetSpec(
        name="TUEV", num_classes=6, is_binary=False,
        splits={"train": "processed_train", "val": "processed_eval", "test": "processed_test"},
        expected_T=1000,
        label_names=("SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"),
    ),
    "Mumtaz": DatasetSpec(
        name="Mumtaz 2016", num_classes=2, is_binary=True,
        splits={"train": "train", "val": "val", "test": "test"},
        expected_T=1000, expected_C=19,
        label_names=("healthy", "depression"),
    ),
    "SHU-MI": DatasetSpec(
        name="SHU-MI", num_classes=2, is_binary=True,
        splits={"train": "train", "val": "val", "test": "test"},
        expected_T=800, expected_C=32,
        label_names=("left", "right"),
    ),
    "Mental-Arithmetic": DatasetSpec(
        name="Mental Arithmetic", num_classes=2, is_binary=True,
        splits={"train": "train", "val": "val", "test": "test"},
        expected_T=1000, expected_C=19,
        label_names=("rest", "task"),
    ),
    "BCIC-IV-2a": DatasetSpec(
        name="BCIC IV 2a", num_classes=4, is_binary=False,
        splits={"train": "train", "val": "val", "test": "test"},
        expected_T=800, expected_C=22,
        label_names=("left", "right", "foot", "tongue"),
    ),
    "SEED-V": DatasetSpec(
        name="SEED-V", num_classes=5, is_binary=False,
        splits={"train": "train", "val": "val", "test": "test"},
        expected_T=200, expected_C=62,
        label_names=("happy", "sad", "neutral", "fear", "disgust"),
    ),
    "BCIC-2020-T3": DatasetSpec(
        name="BCIC 2020-T3", num_classes=5, is_binary=False,
        splits={"train": "train", "val": "val", "test": "test"},
        expected_T=600, expected_C=64,
        label_names=("class_0", "class_1", "class_2", "class_3", "class_4"),
    ),
}


def _label_from_sample(sample) -> int:
    """Best-effort label resolver for diverse pkl conventions."""
    if isinstance(sample, dict):
        for key in ("y", "label", "Y"):
            if key in sample:
                value = sample[key]
                if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                    value = int(np.asarray(value).flatten()[0])
                return int(value)
    raise ValueError("Could not resolve label from pickle sample.")


class InstructionDataset(Dataset):
    """Per-dataset closed-vocab loader (paper §N)."""

    def __init__(self, spec: DatasetSpec, root: str, split: str,
                 ch_names: Optional[Sequence[str]] = None,
                 label_fn: Optional[Callable] = None):
        self.spec = spec
        subdir = spec.splits.get(split, split)
        self.root = Path(root) / subdir
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.files = sorted(self.root.glob("*.pkl"))
        if not self.files:
            raise RuntimeError(f"No .pkl files under {self.root}")
        self.ch_names = (
            list(ch_names) if ch_names
            else list(DEFAULT_TUAB_CHANNELS[: spec.expected_C])
        )
        self.label_fn = label_fn or _label_from_sample

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with open(path, "rb") as handle:
            sample = pickle.load(handle)
        raw = np.asarray(sample["X"], dtype=np.float32)
        T, C = self.spec.expected_T, self.spec.expected_C
        if raw.shape[1] != T:
            raw = raw[:, :T] if raw.shape[1] > T else np.pad(raw, ((0, 0), (0, T - raw.shape[1])))
        if raw.shape[0] != C:
            raw = raw[:C] if raw.shape[0] > C else np.pad(raw, ((0, C - raw.shape[0]), (0, 0)))
        eeg_scaled = torch.from_numpy(raw / 100.0)

        num_time = T // self.spec.patch_size
        time_ids = [t for t in range(num_time) for _ in range(C)]
        chan_ids = get_channel_indices(self.ch_names) * num_time

        patches = rearrange(eeg_scaled, "c (n p) -> (n c) p", p=self.spec.patch_size)
        n_tok = patches.size(0)
        X = torch.zeros((self.spec.block_size, self.spec.patch_size), dtype=torch.float32)
        X[:n_tok] = patches
        chans = torch.full((self.spec.block_size,), PAD_INDEX, dtype=torch.long)
        chans[:n_tok] = torch.tensor(chan_ids, dtype=torch.long)
        times = torch.zeros(self.spec.block_size, dtype=torch.long)
        times[:n_tok] = torch.tensor(time_ids, dtype=torch.long)
        mask = torch.zeros(self.spec.block_size, dtype=torch.bool)
        mask[:n_tok] = True

        descriptors, modality_mask = compute_descriptors(raw, fs=self.spec.fs)

        return {
            "eeg": X,
            "input_chans": chans,
            "input_times": times,
            "input_mask": mask,
            "descriptors": torch.from_numpy(descriptors),
            "modality_mask": torch.from_numpy(modality_mask),
            "stage": torch.tensor(idx / max(1, len(self.files) - 1), dtype=torch.float32),
            "label": int(self.label_fn(sample)),
            "key": path.stem,
        }
