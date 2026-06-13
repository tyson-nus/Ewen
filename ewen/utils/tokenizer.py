"""Helpers for loading the Qwen tokenizer used by the language backbone."""

from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer


def load_qwen_tokenizer(name_or_path: str, cache_dir: Optional[str] = None,
                        use_fast: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        cache_dir=cache_dir,
        use_fast=use_fast,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer
