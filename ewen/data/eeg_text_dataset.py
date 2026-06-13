"""EEG + paired-text dataset used by Stage-1 VQ tokenizer training (paper Appendix B).

Reads windows from a directory of ``.pkl`` files (each holding ``{"X": (C, T), "y": int}``)
and looks the matching text description up in a per-window ``summary_text`` json.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .channels import PAD_INDEX, get_channel_indices

# Canonical 16-channel TUAB unipolar set (used when the pickle file omits ``ch_names``).
DEFAULT_TUAB_CHANNELS: List[str] = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
]


def _norm(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "zscore":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    if mode == "robust":
        median = x.median(dim=-1, keepdim=True).values
        mad = (x - median).abs().median(dim=-1, keepdim=True).values.clamp_min(1e-6)
        return (x - median) / mad
    raise ValueError(f"Unknown norm mode: {mode}")


class EEGTextDataset(Dataset):
    """Pairs an EEG window with a per-window text description.

    Returns
    -------
    X          : (block_size, patch_size) flattened (channel, time-patch) token grid
    Y_raw      : (block_size, patch_size) z-scored reconstruction target
    input_chans: (block_size,) STANDARD_1020 channel id per token
    input_time : (block_size,) time-patch id per token
    input_mask : (block_size,) bool mask for non-padding tokens
    text_ids   : (max_len,) tokenized description
    text_mask  : (max_len,) attention mask
    """

    def __init__(self, data_root: str, meta_json: str, split: str = "train",
                 text_key: str = "summary_text", lm_name: str = "bert-base-uncased",
                 max_len: int = 192, expected_T: int = 2000, expected_C: int = 16,
                 patch_size: int = 200, block_size: int = 1024, norm: str = "robust",
                 ch_names: Optional[Sequence[str]] = None):
        self.data_root = Path(data_root) / split
        if not self.data_root.exists():
            raise FileNotFoundError(f"split directory does not exist: {self.data_root}")
        self.files = sorted(self.data_root.glob("*.pkl"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pkl files found under {self.data_root}")

        self.text_key = text_key
        self.max_len = int(max_len)
        self.expected_T = int(expected_T)
        self.expected_C = int(expected_C)
        self.patch_size = int(patch_size)
        self.block_size = int(block_size)
        self.norm = norm
        self.ch_names = list(ch_names) if ch_names else list(DEFAULT_TUAB_CHANNELS[: self.expected_C])

        self.meta = self._load_meta(meta_json)
        self.files = [f for f in self.files if self._key(f) in self.meta]
        if len(self.files) == 0:
            raise RuntimeError(f"No EEG file in {self.data_root} matched any key of {meta_json}")

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=False, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

    @staticmethod
    def _key(path: Path) -> str:
        return path.stem

    def _load_meta(self, meta_json: str):
        if not os.path.exists(meta_json):
            raise FileNotFoundError(meta_json)
        with open(meta_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("meta_json must be a JSON object mapping {window_key: text|payload}")
        out = {}
        for key, value in data.items():
            text = value if isinstance(value, str) else value.get(self.text_key) if isinstance(value, dict) else None
            if isinstance(text, str) and text:
                out[str(key)] = text
        return out

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with open(path, "rb") as handle:
            sample = pickle.load(handle)
        data = np.asarray(sample["X"], dtype=np.float32)
        if data.shape[1] > self.expected_T:
            data = data[:, : self.expected_T]
        elif data.shape[1] < self.expected_T:
            data = np.pad(data, ((0, 0), (0, self.expected_T - data.shape[1])))
        if data.shape[0] != self.expected_C:
            if data.shape[0] > self.expected_C:
                data = data[: self.expected_C]
            else:
                data = np.pad(data, ((0, self.expected_C - data.shape[0]), (0, 0)))
        data = torch.from_numpy(data / 100.0)  # mild amplitude scaling

        num_time_patches = self.expected_T // self.patch_size
        time_ids = [t for t in range(num_time_patches) for _ in range(self.expected_C)]
        chan_ids = get_channel_indices(self.ch_names) * num_time_patches

        # (C, num_patches*patch) -> (num_patches*C, patch)
        data_p = rearrange(data, "c (n p) -> (n c) p", p=self.patch_size)
        num_tokens = data_p.size(0)
        assert num_tokens <= self.block_size, "block_size too small for the (C, T_patches) layout"

        X = torch.zeros((self.block_size, self.patch_size), dtype=torch.float32)
        Y_raw = torch.zeros((self.block_size, self.patch_size), dtype=torch.float32)
        X[:num_tokens] = data_p
        Y_raw[:num_tokens] = _norm(data_p, self.norm)

        input_chans = torch.full((self.block_size,), PAD_INDEX, dtype=torch.long)
        input_chans[:num_tokens] = torch.tensor(chan_ids, dtype=torch.long)
        input_time = torch.zeros(self.block_size, dtype=torch.long)
        input_time[:num_tokens] = torch.tensor(time_ids, dtype=torch.long)
        input_mask = torch.zeros(self.block_size, dtype=torch.bool)
        input_mask[:num_tokens] = True

        text = self.meta[self._key(path)]
        tok = self.tokenizer(text, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=self.max_len)
        text_ids = tok["input_ids"].squeeze(0)
        text_mask = tok["attention_mask"].squeeze(0)
        return X, Y_raw, input_chans, input_time, input_mask, text_ids, text_mask
