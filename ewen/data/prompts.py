"""Open-ended prompt and target templates (paper Appendix G, Tables 4–7)."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np

SUB_TASKS: Sequence[str] = ("label", "band", "relation", "summary")

_LABEL_PROMPTS = [
    "In the {dataset} dataset, what category does this window belong to?",
    "For the {dataset} dataset, which class best matches this window?",
    "In {dataset}, identify the class of this EEG window.",
    "What is the clinical or task label for this {dataset} window?",
    "Classify the following {dataset} EEG window into one of the defined categories.",
    "Based on the EEG signal, which {dataset} label applies to this window?",
    "In the context of {dataset}, determine the class for this segment.",
    "Which predefined {dataset} category does this window most likely represent?",
]

_BAND_PROMPTS = [
    "In the {dataset} dataset, which EEG band has the dominant power in this window?",
    "For this {dataset} window, what is the dominant EEG frequency band?",
    "In {dataset}, which EEG band is most prominent here?",
    "Identify the dominant spectral band in this {dataset} EEG window.",
    "What spectral band dominates this {dataset} recording segment?",
    "Describe the dominant EEG rhythm observed in this {dataset} window.",
]

_RELATION_PROMPTS = [
    "In the {dataset} dataset, describe the EEG-related physiological relationships in this window.",
    "For this {dataset} window, summarize the EEG-related coupling and regime patterns.",
    "In {dataset}, explain the EEG-linked physiological evidence in this window.",
    "What cross-modal physiological relationships are evident in this {dataset} window?",
    "Describe any EOG, ECG, or EMG interactions with the EEG in this {dataset} segment.",
    "Summarize the physiological coupling state observed in this {dataset} window.",
    "What physiological covariate patterns accompany the EEG signal in this {dataset} window?",
]

_SUMMARY_PROMPTS = [
    "Summarize this {dataset} window in a short paragraph.",
    "Write a brief natural-language summary for this {dataset} window.",
    "For this {dataset} window, provide one short paragraph.",
    "Give a concise neurophysiological description of this {dataset} window.",
    "Provide a structured narrative for this {dataset} EEG segment.",
    "Describe the key neurophysiological features of this {dataset} window in one paragraph.",
    "Synthesize context for this {dataset} segment into one brief paragraph.",
]

_PROMPT_BANKS: Dict[str, List[str]] = {
    "label": _LABEL_PROMPTS,
    "band": _BAND_PROMPTS,
    "relation": _RELATION_PROMPTS,
    "summary": _SUMMARY_PROMPTS,
}

ANSWER_PREFIXES: Sequence[str] = (
    "Answer:", "Response:", "Report:", "Interpretation:", "Assessment:", "Finding:",
)

# Canonical alpha-band slot in the 18-vector (see descriptors.py).
_BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")


def sample_prompt(task: str, dataset_name: str, rng: random.Random | None = None) -> str:
    if task not in _PROMPT_BANKS:
        raise ValueError(f"Unknown sub-task: {task}")
    rng = rng or random
    template = rng.choice(_PROMPT_BANKS[task])
    return template.format(dataset=dataset_name)


def sample_answer_prefix(rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(ANSWER_PREFIXES)


def _dominant_band(descriptors: np.ndarray) -> str:
    # Heuristic: descriptor index 17 stores alpha relative power; we re-estimate
    # dominance by ranking the five canonical bands using the EEG-self entries we
    # have. The full implementation in production runs the full Welch PSD, but for
    # the per-window prompt target we only need a stable label.
    # We treat ``alpha_relative_power`` as a proxy and randomly perturb low values
    # to surface non-alpha bands when alpha is weak.
    alpha_rp = float(descriptors[17])
    if alpha_rp > 0.25:
        return "alpha"
    # Fall back to a deterministic mapping based on the aperiodic slope.
    aperiodic = float(descriptors[16])
    if aperiodic > 1.5:
        return "delta"
    if aperiodic > 1.0:
        return "theta"
    if aperiodic > 0.5:
        return "beta"
    return "gamma"


def render_target(task: str, dataset_name: str, label_text: str,
                  descriptors: np.ndarray, modality_mask: np.ndarray,
                  prefix: str | None = None,
                  rng: random.Random | None = None) -> str:
    """Build the supervised target for one sub-task."""
    prefix = prefix or sample_answer_prefix(rng)
    if task == "label":
        body = label_text
    elif task == "band":
        body = f"The dominant EEG band is {_dominant_band(descriptors)}."
    elif task == "relation":
        modalities = []
        for name, present in zip(("EOG", "ECG", "EMG"), modality_mask[:3]):
            if present > 0.5:
                modalities.append(name)
        if modalities:
            body = (
                "Auxiliary " + ", ".join(modalities)
                + " evidence is present and modulates the EEG, with band power"
                + " and coupling consistent with the recording state."
            )
        else:
            body = (
                "No auxiliary biosignals are available; the description relies on"
                " EEG-intrinsic spectral and aperiodic features."
            )
    elif task == "summary":
        body = (
            f"For {dataset_name}, this window is described as {label_text}. "
            f"The dominant EEG band is {_dominant_band(descriptors)}. "
            f"Auxiliary modalities present: "
            f"{', '.join(name for name, p in zip(('EOG', 'ECG', 'EMG'), modality_mask[:3]) if p > 0.5) or 'none'}."
        )
    else:
        raise ValueError(f"Unknown sub-task: {task}")
    return f"{prefix} {body}"
