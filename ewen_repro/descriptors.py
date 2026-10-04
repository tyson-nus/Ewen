from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

DESCRIPTOR_NAMES = (
    "eeg_relative_delta", "eeg_relative_theta", "eeg_relative_alpha",
    "eeg_relative_beta", "eeg_relative_gamma", "eeg_alpha_frequency",
    "eeg_aperiodic_exponent", "eog_blink_rate", "eog_saccade_rate",
    "eog_median_spv", "eog_eeg_max_lagged_correlation",
    "eog_eeg_regression_gain", "ecg_mean_heart_rate", "ecg_sdnn",
    "ecg_rmssd", "ecg_eeg_hep_amplitude", "ecg_eeg_plv", "emg_rms",
    "emg_median_frequency", "emg_spectral_entropy", "emg_eeg_cmc",
    "emg_eeg_phase_slope_delay",
)
MODALITY_NAMES = ("EOG", "ECG", "EMG")
AUXILIARY_SPECS = (
    ("EOG", "Blink Rate"), ("EOG", "Saccade Rate"),
    ("EOG", "Median Saccadic Peak Velocity"),
    ("EOG-EEG", "Max Lagged Correlation"),
    ("EOG-EEG", "EOG to EEG Regression Gain"),
    ("ECG", "Mean Heart Rate"), ("ECG", "SDNN"), ("ECG", "RMSSD"),
    ("ECG-EEG", "Heartbeat Evoked Potential Amplitude"),
    ("ECG-EEG", "Cardio-Cerebral Phase Locking Value"),
    ("EMG", "Root Mean Square Amplitude"), ("EMG", "Median Frequency"),
    ("EMG", "Spectral Entropy"),
    ("EMG-EEG", "Corticomuscular Coherence"),
    ("EMG-EEG", "Coherence Phase-Slope Delay"),
)
_AUXILIARY_ALIASES = {
    "Root Mean Square Amplitude": ("Root Mean Square",),
    "Coherence Phase-Slope Delay": ("Coherence Phase Slope Delay",),
}


LEGACY_NONCONFORMING_INDICES = (10, 15, 18, 19, 21)


@dataclass(frozen=True)
class DescriptorConfig:

    bands: tuple = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    normalization_band: tuple = (0.5, 45.0)
    welch_seconds: float = 2.0
    welch_overlap_fraction: float = 0.5
    aperiodic_band: tuple = (2.0, 40.0)
    aperiodic_exclude_bands: tuple = ((8.0, 13.0),)
    alpha_filter_order: int = 4
    alpha_edge_trim_seconds: float = 0.0
    algorithm_revision: str = "appendix_f_v2"

    def __post_init__(self):
        if len(self.bands) != 5 or any(not 0 <= low < high for low, high in self.bands):
            raise ValueError("The descriptor layout requires five ordered frequency bands")
        if self.welch_seconds <= 0 or not 0 <= self.welch_overlap_fraction < 1:
            raise ValueError("Invalid Welch configuration")
        if not 0 <= self.normalization_band[0] < self.normalization_band[1]:
            raise ValueError("Invalid relative-power normalization band")
        if self.alpha_edge_trim_seconds < 0:
            raise ValueError("Alpha edge trimming must be nonnegative")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _integral(freqs: np.ndarray, values: np.ndarray, low: float, high: float) -> float:

    low, high = max(low, float(freqs[0])), min(high, float(freqs[-1]))
    if high <= low:
        return 0.0
    selected = (freqs > low) & (freqs < high)
    f = np.concatenate(([low], freqs[selected], [high]))
    p = np.concatenate(([np.interp(low, freqs, values)], values[selected],
                        [np.interp(high, freqs, values)]))
    return float(trapezoid(p, f))


def eeg_descriptors(eeg: np.ndarray, sfreq: float, config: DescriptorConfig | None = None):

    cfg = config or DescriptorConfig()
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or sfreq <= 0:
        raise ValueError("EEG must be [channels, time] with a positive sampling rate")
    out = np.zeros(7, dtype=np.float32)
    observed = np.zeros(7, dtype=bool)
    good = np.all(np.isfinite(eeg), axis=1) & (np.std(eeg, axis=1) > 1e-12)
    x = eeg[good]
    if not len(x) or x.shape[1] < 8:
        return out, observed
    nperseg = min(x.shape[1], max(8, int(round(cfg.welch_seconds * sfreq))))
    freqs, psds = signal.welch(x, fs=sfreq, nperseg=nperseg,
                              noverlap=int(nperseg * cfg.welch_overlap_fraction),
                              detrend="constant", axis=-1)
    psd = psds.mean(axis=0)
    denominator = _integral(freqs, psd, *cfg.normalization_band)
    if denominator > 1e-20 and cfg.normalization_band[1] <= sfreq / 2:
        for i, (low, high) in enumerate(cfg.bands):
            if high <= sfreq / 2:
                out[i] = _integral(freqs, psd, low, high) / denominator
                observed[i] = True
    if sfreq / 2 > cfg.bands[2][1]:
        sos = signal.butter(cfg.alpha_filter_order, cfg.bands[2], btype="bandpass",
                            fs=sfreq, output="sos")
        try:
            alpha = signal.sosfiltfilt(sos, x, axis=-1)
            analytic = signal.hilbert(alpha, axis=-1)
            inst = np.diff(np.unwrap(np.angle(analytic), axis=-1), axis=-1) * sfreq / (2 * np.pi)


            trim = min(int(cfg.alpha_edge_trim_seconds * sfreq),
                       max(0, (inst.shape[-1] - 4) // 2))
            if trim:
                inst = inst[:, trim:-trim]
            value = float(inst.mean())
            if np.isfinite(value):
                out[5], observed[5] = value, True
        except ValueError:
            pass
    use = (freqs >= cfg.aperiodic_band[0]) & (freqs <= cfg.aperiodic_band[1]) & (psd > 1e-20)
    for low, high in cfg.aperiodic_exclude_bands:
        use &= ~((freqs >= low) & (freqs <= high))
    if use.sum() >= 3 and cfg.aperiodic_band[1] <= sfreq / 2:
        slope = np.polyfit(np.log(freqs[use]), np.log(psd[use]), 1)[0]
        if np.isfinite(slope):
            out[6], observed[6] = -float(slope), True
    return out, observed


def descriptor_vector(eeg: np.ndarray, sfreq: float, auxiliary: Mapping | None = None,
                      config: DescriptorConfig | None = None, auxiliary_policy="provided"):

    if auxiliary_policy not in ("provided", "audited_legacy", "raw"):
        raise ValueError("Unsupported auxiliary descriptor policy")
    values = np.zeros(22, dtype=np.float32)
    mask = np.zeros(22, dtype=bool)
    values[:7], mask[:7] = eeg_descriptors(eeg, sfreq, config)
    sidecar = auxiliary or {}
    metrics = sidecar.get("metrics", {})
    availability = sidecar.get("modality_presence", {})
    modalities = np.array([availability.get(m) is True for m in MODALITY_NAMES], dtype=bool)
    for i, (group, name) in enumerate(AUXILIARY_SPECS, start=7):
        modality = group.split("-")[0]
        if not modalities[MODALITY_NAMES.index(modality)]:
            continue
        block = metrics.get(group, {})
        value = block.get(name)
        if value is None:
            for alias in _AUXILIARY_ALIASES.get(name, ()):
                if alias in block:
                    value = block[alias]
                    break

        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and abs(value) <= float(np.finfo(np.float32).max)):
            values[i], mask[i] = float(value), True
    if auxiliary_policy == "audited_legacy":

        values[13:15] *= 0.001
        values[list(LEGACY_NONCONFORMING_INDICES)] = 0
        mask[list(LEGACY_NONCONFORMING_INDICES)] = False
    return values, mask, modalities


class DescriptorStandardizer:

    def __init__(self):
        self.count = np.zeros(22, dtype=np.int64)
        self.mean = np.zeros(22, dtype=np.float64)
        self.m2 = np.zeros(22, dtype=np.float64)
        self.fitted = False
        self.source_split = None
        self.sample_count = 0
        self.extractor_fingerprint = None

    @property
    def std(self):

        var = self.m2 / np.maximum(self.count, 1)
        return np.where(var > 1e-12, np.sqrt(var), 1.0)

    def fit(self, samples: Iterable, split: str = "train", extractor_fingerprint: str | None = None):
        if split != "train":
            raise ValueError("Descriptor statistics can only be fitted from train")
        if self.fitted or self.sample_count:
            raise ValueError("Create a new standardizer to fit another training corpus")
        for item in samples:
            if isinstance(item, Mapping):
                if item.get("split", "train") != "train":
                    raise ValueError("Validation/test samples cannot fit descriptor statistics")
                values, mask = item["descriptors"], item["descriptor_mask"]
            else:
                values, mask = item[:2]
            values = np.asarray(values, dtype=np.float64)
            observed = np.asarray(mask, dtype=bool) & np.isfinite(values)
            if values.shape != (22,) or observed.shape != (22,):
                raise ValueError("Descriptor values and masks must each have length 22")
            self.count[observed] += 1
            delta = values[observed] - self.mean[observed]
            self.mean[observed] += delta / self.count[observed]
            self.m2[observed] += delta * (values[observed] - self.mean[observed])
            self.sample_count += 1
        if not self.sample_count:
            raise ValueError("Cannot fit an empty training split")
        self.source_split, self.fitted = "train", True
        self.extractor_fingerprint = extractor_fingerprint
        return self

    def transform(self, values, mask):
        if not self.fitted or self.source_split != "train":
            raise ValueError("Load or fit train-only descriptor statistics before transforming")
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool) & (self.count > 0) & np.isfinite(values)
        if values.shape[-1:] != (22,):
            raise ValueError("Descriptor values must have 22 dimensions")
        scaled = (values - self.mean) / self.std
        result = np.where(mask, scaled, 0).astype(np.float32)
        return result, mask

    def state_dict(self):
        return {"schema": 1, "names": list(DESCRIPTOR_NAMES), "source_split": self.source_split,
                "sample_count": self.sample_count, "count": self.count.tolist(),
                "mean": self.mean.tolist(), "m2": self.m2.tolist(),
                "extractor_fingerprint": self.extractor_fingerprint, "fitted": self.fitted}

    def save(self, path):
        if not self.fitted:
            raise ValueError("Cannot save unfitted descriptor statistics")
        Path(path).write_text(json.dumps(self.state_dict(), indent=2) + "\n")

    @classmethod
    def from_state_dict(cls, state):
        if state.get("source_split") != "train" or state.get("names") != list(DESCRIPTOR_NAMES):
            raise ValueError("Incompatible or non-training descriptor statistics")
        obj = cls()
        for key, dtype in (("count", np.int64), ("mean", np.float64), ("m2", np.float64)):
            value = np.asarray(state[key], dtype=dtype)
            if value.shape != (22,) or not np.all(np.isfinite(value)):
                raise ValueError("Invalid descriptor statistics")
            setattr(obj, key, value)
        if np.any(obj.count < 0) or np.any(obj.m2 < -1e-10):
            raise ValueError("Invalid descriptor counts or variance")
        obj.source_split = "train"
        obj.sample_count = int(state["sample_count"])
        obj.extractor_fingerprint = state.get("extractor_fingerprint")
        obj.fitted = bool(state.get("fitted"))
        if not obj.fitted or obj.sample_count <= 0:
            raise ValueError("Statistics have not been fitted")
        return obj

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text()))


def descriptor_text(values, mask):

    return "; ".join(f"{name}={float(value):.4g}" for name, value, seen in
                     zip(DESCRIPTOR_NAMES, values, mask) if seen) + "."
