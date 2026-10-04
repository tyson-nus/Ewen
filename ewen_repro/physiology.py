from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid, trapezoid


@dataclass(frozen=True)
class PhysiologyConfig:
    representative_eeg_channel: int = 0
    blink_band: tuple = (0.1, 10.0)
    saccade_band: tuple = (0.1, 15.0)
    r_peak_band: tuple = (5.0, 20.0)
    blink_threshold_mad: float = 4.0
    saccade_threshold_mad: float = 4.0
    r_peak_threshold_mad: float = 4.0
    blink_minimum_separation_seconds: float = 0.3
    saccade_minimum_separation_seconds: float = 0.2
    r_peak_minimum_separation_seconds: float = 0.3
    maximum_lag_seconds: float = 0.2
    hep_interval_seconds: tuple = (0.25, 0.55)
    plv_eeg_band: tuple | None = None
    plv_ecg_band: tuple | None = None
    welch_seconds: float = 2.0
    beta_band: tuple = (13.0, 30.0)
    rr_sd_ddof: int = 1
    filter_order: int = 4

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _one_channel(x, length, name):
    if x is None:
        return None
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(f"{name} requires an explicitly selected single channel")
        values = values[0]
    if values.ndim != 1 or len(values) != length:
        raise ValueError(f"{name} and EEG must have identical aligned window lengths")
    if not np.isfinite(values).all() or np.std(values) <= 1e-12:
        return None
    return values


def _filter(x, sfreq, band, config):
    if band is None:
        return x
    if not 0 < band[0] < band[1] < sfreq / 2:
        raise ValueError("Configured filter band exceeds the signal Nyquist range")
    sos = signal.butter(config.filter_order, band, fs=sfreq, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def _peaks(x, sfreq, separation, threshold_mad):
    amplitude = np.abs(x - np.median(x))
    center = np.median(amplitude)
    scale = 1.4826 * np.median(np.abs(amplitude - center))
    threshold = center + threshold_mad * max(scale, 1e-12)
    peaks, _ = signal.find_peaks(amplitude, height=threshold,
                                 distance=max(1, int(round(sfreq * separation))))
    return peaks


def maximum_signed_lagged_correlation(x, y, sfreq, maximum_lag_seconds=0.2):

    limit = int(round(sfreq * maximum_lag_seconds))
    values = []
    for lag in range(-limit, limit + 1):
        if lag < 0:
            a, b = x[-lag:], y[:len(y) + lag]
        elif lag > 0:
            a, b = x[:-lag], y[lag:]
        else:
            a, b = x, y
        if len(a) >= 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
            value = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(value):
                values.append(value)
    return max(values) if values else float("nan")


def rr_metrics(r_peaks, sfreq, ddof=1):

    rr = np.diff(np.asarray(r_peaks, dtype=float)) / sfreq
    if not len(rr) or np.any(rr <= 0):
        return {}
    result = {"Mean Heart Rate": float(60 / rr.mean())}
    if len(rr) > ddof:
        result["SDNN"] = float(np.std(rr, ddof=ddof))
    if len(rr) >= 2:
        result["RMSSD"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    return result


def heartbeat_evoked_potential(eeg, r_peaks, sfreq, interval=(0.25, 0.55)):

    low, high = interval
    if not 0 <= low < high:
        raise ValueError("HEP endpoints must be nonnegative and increasing")
    times = np.arange(len(eeg)) / sfreq
    means = []
    for peak in r_peaks:
        start, stop = peak / sfreq + low, peak / sfreq + high
        if start < times[0] or stop > times[-1]:
            continue
        interior = (times > start) & (times < stop)
        t = np.concatenate(([start], times[interior], [stop]))
        y = np.concatenate(([np.interp(start, times, eeg)], eeg[interior],
                            [np.interp(stop, times, eeg)]))
        means.append(float(trapezoid(y, t) / (high - low)))
    return float(np.mean(means)) if means else float("nan")


def emg_self_metrics(emg, sfreq, welch_seconds=2.0):

    nperseg = min(len(emg), max(8, int(round(sfreq * welch_seconds))))
    f, p = signal.welch(emg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    area = cumulative_trapezoid(p, f, initial=0)
    result = {"Root Mean Square Amplitude": float(np.sqrt(np.mean(emg ** 2)))}
    if area[-1] > 1e-20:
        result["Median Frequency"] = float(np.interp(area[-1] / 2, area, f))
    if p.sum() > 1e-20:
        probabilities = p / p.sum()
        positive = probabilities > 0
        result["Spectral Entropy"] = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return result


def eeg_emg_coupling(eeg, emg, sfreq, config=None):

    cfg = config or PhysiologyConfig()
    nperseg = min(len(eeg), max(8, int(round(sfreq * cfg.welch_seconds))))
    settings = dict(fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    freqs, coherence = signal.coherence(eeg, emg, **settings)
    _, cross_scipy = signal.csd(eeg, emg, **settings)

    cross = np.conj(cross_scipy)
    _, pxx = signal.welch(eeg, **settings)
    _, pyy = signal.welch(emg, **settings)
    low, high = cfg.beta_band
    in_band = (freqs >= low) & (freqs <= high)
    result = {}
    if in_band.sum() >= 2 and np.isfinite(coherence[in_band]).all():
        inner = (freqs > low) & (freqs < high)
        f = np.concatenate(([low], freqs[inner], [high]))
        c = np.concatenate(([np.interp(low, freqs, coherence)], coherence[inner],
                            [np.interp(high, freqs, coherence)]))
        result["Corticomuscular Coherence"] = float(trapezoid(c, f) / (high - low))
    valid = in_band & np.isfinite(cross) & (pxx > 1e-20) & (pyy > 1e-20) & (np.abs(cross) > 1e-20)
    if valid.sum() >= 2:
        phase = np.unwrap(np.angle(cross[valid]))
        slope = np.polyfit(freqs[valid], phase, 1)[0]
        result["Coherence Phase-Slope Delay"] = float(-slope / (2 * np.pi))
    return result


def compute_auxiliary_metrics(eeg, sfreq, eog=None, ecg=None, emg=None,
                              eog_diff=None, config=None):

    cfg = config or PhysiologyConfig()
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim == 1:
        eeg = eeg[None]
    if eeg.ndim != 2 or eeg.shape[1] < 8 or sfreq <= 0:
        raise ValueError("EEG must be a valid [channel,time] window")
    if not 0 <= cfg.representative_eeg_channel < eeg.shape[0]:
        raise ValueError("Representative EEG channel is outside the window")
    reference = eeg[cfg.representative_eeg_channel]
    usable_eeg = np.isfinite(reference).all() and np.std(reference) > 1e-12
    n = eeg.shape[1]
    eog, ecg, emg = (_one_channel(x, n, name) for x, name in ((eog, "EOG"), (ecg, "ECG"), (emg, "EMG")))
    diff = _one_channel(eog_diff, n, "differential EOG") if eog_diff is not None else eog
    metrics = {}
    presence = {"EEG": usable_eeg, "EOG": eog is not None, "ECG": ecg is not None, "EMG": emg is not None}
    if eog is not None:
        blink_signal = _filter(eog, sfreq, cfg.blink_band, cfg)
        blinks = _peaks(blink_signal, sfreq, cfg.blink_minimum_separation_seconds, cfg.blink_threshold_mad)
        block = {"Blink Rate": float(len(blinks) * 60 * sfreq / n)}
        if diff is not None:
            velocity = np.gradient(_filter(diff, sfreq, cfg.saccade_band, cfg)) * sfreq
            saccades = _peaks(velocity, sfreq, cfg.saccade_minimum_separation_seconds, cfg.saccade_threshold_mad)
            block["Saccade Rate"] = float(len(saccades) * 60 * sfreq / n)
            if len(saccades):
                block["Median Saccadic Peak Velocity"] = float(np.median(np.abs(velocity[saccades])))
        metrics["EOG"] = block
        if usable_eeg:
            metrics["EOG-EEG"] = {"Max Lagged Correlation": maximum_signed_lagged_correlation(eog, reference, sfreq, cfg.maximum_lag_seconds),
                                  "EOG to EEG Regression Gain": float(np.dot(reference, eog) / (np.dot(eog, eog) + 1e-12))}
    if ecg is not None:
        r_peaks = _peaks(_filter(ecg, sfreq, cfg.r_peak_band, cfg), sfreq,
                         cfg.r_peak_minimum_separation_seconds, cfg.r_peak_threshold_mad)
        metrics["ECG"] = rr_metrics(r_peaks, sfreq, cfg.rr_sd_ddof)
        if usable_eeg:
            phase_eeg = np.angle(signal.hilbert(_filter(reference, sfreq, cfg.plv_eeg_band, cfg)))
            phase_ecg = np.angle(signal.hilbert(_filter(ecg, sfreq, cfg.plv_ecg_band, cfg)))
            metrics["ECG-EEG"] = {"Heartbeat Evoked Potential Amplitude": heartbeat_evoked_potential(reference, r_peaks, sfreq, cfg.hep_interval_seconds),
                                  "Cardio-Cerebral Phase Locking Value": float(np.abs(np.mean(np.exp(1j * (phase_eeg - phase_ecg)))))}
    if emg is not None:
        metrics["EMG"] = emg_self_metrics(emg, sfreq, cfg.welch_seconds)
        if usable_eeg:
            metrics["EMG-EEG"] = eeg_emg_coupling(reference, emg, sfreq, cfg)

    metrics = {group: {key: float(value) for key, value in values.items() if np.isfinite(value)}
               for group, values in metrics.items()}
    return {"metrics": metrics, "modality_presence": presence, "provenance": "aligned_raw_auxiliary_signals",
            "configuration": asdict(cfg), "configuration_sha256": cfg.fingerprint,
            "units": {"SDNN": "seconds", "RMSSD": "seconds", "HEP": "provided EEG amplitude unit",
                      "EMG RMS": "provided EMG amplitude unit", "phase_delay": "seconds"}}
