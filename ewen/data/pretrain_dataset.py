"""Stage-2 pretraining dataset (paper Appendix N).

Yields (EEG window + paired text description). Descriptors are computed online from
the EEG signal so the dataset is self-contained.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from .channels import PAD_INDEX, get_channel_indices
from .descriptors import DESCRIPTOR_DIM, NUM_MODALITIES, compute_descriptors
from .eeg_text_dataset import DEFAULT_TUAB_CHANNELS


class PretrainEEGDataset(Dataset):
    """Pickled-EEG corpus paired with descriptor text from a per-window summary json."""

    def __init__(self, data_root: str, meta_json: str, split: str = "train",
                 text_key: str = "summary_text", expected_T: int = 2000,
                 expected_C: int = 16, patch_size: int = 200, block_size: int = 1024,
                 fs: int = 200, ch_names: Optional[Sequence[str]] = None,
                 max_text_len: int = 192):
        self.data_root = Path(data_root) / split
        self.files = sorted(self.data_root.glob("*.pkl"))
        if not self.files:
            raise RuntimeError(f"No .pkl files under {self.data_root}")
        self.text_key = text_key
        self.expected_T = int(expected_T)
        self.expected_C = int(expected_C)
        self.patch_size = int(patch_size)
        self.block_size = int(block_size)
        self.fs = int(fs)
        self.ch_names = list(ch_names) if ch_names else list(DEFAULT_TUAB_CHANNELS[: self.expected_C])
        self.max_text_len = int(max_text_len)

        if not os.path.exists(meta_json):
            raise FileNotFoundError(meta_json)
        with open(meta_json, "r", encoding="utf-8") as handle:
            meta_all = json.load(handle)
        if not isinstance(meta_all, dict):
            raise ValueError("meta_json must be a dict")
        self.text_by_key = {}
        for key, value in meta_all.items():
            text = value if isinstance(value, str) else value.get(text_key) if isinstance(value, dict) else None
            if isinstance(text, str) and text:
                self.text_by_key[str(key)] = text
        self.files = [f for f in self.files if f.stem in self.text_by_key]
        if not self.files:
            raise RuntimeError("No pkl/meta-json overlap found.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with open(path, "rb") as handle:
            sample = pickle.load(handle)
        raw = np.asarray(sample["X"], dtype=np.float32)
        if raw.shape[1] > self.expected_T:
            raw = raw[:, : self.expected_T]
        elif raw.shape[1] < self.expected_T:
            raw = np.pad(raw, ((0, 0), (0, self.expected_T - raw.shape[1])))
        if raw.shape[0] != self.expected_C:
            if raw.shape[0] > self.expected_C:
                raw = raw[: self.expected_C]
            else:
                raw = np.pad(raw, ((0, self.expected_C - raw.shape[0]), (0, 0)))
        eeg_scaled = torch.from_numpy(raw / 100.0)  # for VQ input

        num_time = self.expected_T // self.patch_size
        time_ids = [t for t in range(num_time) for _ in range(self.expected_C)]
        chan_ids = get_channel_indices(self.ch_names) * num_time

        patches = rearrange(eeg_scaled, "c (n p) -> (n c) p", p=self.patch_size)
        n_tok = patches.size(0)
        X = torch.zeros((self.block_size, self.patch_size), dtype=torch.float32)
        X[:n_tok] = patches
        chans = torch.full((self.block_size,), PAD_INDEX, dtype=torch.long)
        chans[:n_tok] = torch.tensor(chan_ids, dtype=torch.long)
        times = torch.zeros(self.block_size, dtype=torch.long)
        times[:n_tok] = torch.tensor(time_ids, dtype=torch.long)
        mask = torch.zeros(self.block_size, dtype=torch.bool)
        mask[:n_tok] = True

        descriptors, modality_mask = compute_descriptors(raw, fs=self.fs)
        descriptors = torch.from_numpy(descriptors)
        modality_mask = torch.from_numpy(modality_mask)
        stage = torch.tensor(idx / max(1, len(self.files) - 1), dtype=torch.float32)

        text = self.text_by_key[path.stem]
        label = int(sample.get("y", 0))

        return {
            "eeg": X,
            "input_chans": chans,
            "input_times": times,
            "input_mask": mask,
            "descriptors": descriptors,
            "modality_mask": modality_mask,
            "stage": stage,
            "text": text,
            "label": label,
            "key": path.stem,
        }
