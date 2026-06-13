"""Descriptorized physiological covariates (paper Appendix F, Eq. 1).

Computes 18 scalar descriptors organised in five modality families:
    EOG self-state (3) + EOG-EEG coupling (2) +
    ECG self-state (3) + ECG-EEG coupling (2) +
    EMG self-state (3) + EMG-EEG coupling (2) +
    EEG self-state (3).

Layout (canonical order, matches Appendix F):
    [blink_rate, saccade_rate, sacc_peak_v,                  # EOG self
     eog_eeg_corr_max, eog_eeg_reg_gain,                     # EOG-EEG
     hr_mean, sdnn, rmssd,                                   # ECG self
     hep_amp, plv_eeg_ecg,                                   # ECG-EEG
     emg_rms, emg_mdf, emg_spec_entropy,                     # EMG self
     cmc_beta, cmc_phase_slope,                              # EMG-EEG
     mean_alpha_freq, aperiodic_slope, alpha_relative_power] # EEG self

If a modality is unobserved, its corresponding entries are masked to zero per the
missing-modality fallback described in paper §3.1 and Appendix F. Returned alongside the
18-vector is a 4-bit modality-presence mask ``δ ∈ {0,1}^4`` for [EOG, ECG, EMG, EEG].
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import coherence, hilbert, welch

DESCRIPTOR_DIM = 18
NUM_MODALITIES = 4  # EOG, ECG, EMG, EEG


_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _safe_float(value: float, default: float = 0.0) -> float:
    if not np.isfinite(value):
        return default
    return float(value)


def _band_power(psd_freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (psd_freqs >= lo) & (psd_freqs < hi)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], psd_freqs[mask]))


def _detect_peaks(signal: np.ndarray, fs: float, min_distance: float = 0.4,
                  threshold_factor: float = 3.0) -> np.ndarray:
    """Very simple amplitude-threshold peak detector (no SciPy dependency)."""
    if signal.size < 3:
        return np.zeros(0, dtype=np.int64)
    threshold = np.median(np.abs(signal - np.median(signal))) * threshold_factor
    threshold = max(threshold, 1e-6)
    min_gap = int(min_distance * fs)
    candidates = np.where(
        (signal[1:-1] > signal[:-2])
        & (signal[1:-1] >= signal[2:])
        & (signal[1:-1] > threshold)
    )[0] + 1
    if candidates.size == 0:
        return candidates
    keep = [candidates[0]]
    for idx in candidates[1:]:
        if idx - keep[-1] >= min_gap:
            keep.append(idx)
    return np.asarray(keep, dtype=np.int64)


def _bandpass(signal: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """FIR-free band-pass via FFT (lightweight, dependency-minimal)."""
    n = signal.size
    if n == 0:
        return signal
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    fft[~mask] = 0.0
    return np.fft.irfft(fft, n=n)


def _eeg_self_descriptors(eeg: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """Returns (mean instantaneous alpha freq, aperiodic slope, alpha relative power)."""
    if eeg.ndim == 2:
        x = eeg.mean(axis=0)
    else:
        x = eeg
    if x.size < int(fs * 0.5):
        return 0.0, 0.0, 0.0
    nperseg = min(x.size, int(fs * 2))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    total = _band_power(freqs, psd, freqs[1], freqs[-1])
    if total <= 0:
        return 0.0, 0.0, 0.0
    alpha_rp = _band_power(freqs, psd, *_BANDS["alpha"]) / total
    alpha_filtered = _bandpass(x, fs, *_BANDS["alpha"])
    analytic = hilbert(alpha_filtered)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    mean_alpha_freq = _safe_float(np.nanmean(inst_freq))
    log_f = np.log(freqs[1:])
    log_p = np.log(np.clip(psd[1:], 1e-12, None))
    if log_f.size >= 2:
        slope, _ = np.polyfit(log_f, log_p, 1)
        aperiodic = _safe_float(-slope)
    else:
        aperiodic = 0.0
    return mean_alpha_freq, aperiodic, alpha_rp


def _eog_self_descriptors(eog: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """Blink rate (per minute), saccade rate, median saccadic peak velocity."""
    x = eog[0] if eog.ndim == 2 else eog
    duration_minutes = x.size / fs / 60.0
    if duration_minutes <= 0:
        return 0.0, 0.0, 0.0
    blinks = _detect_peaks(np.abs(x), fs, min_distance=0.3, threshold_factor=5.0)
    blink_rate = float(blinks.size) / duration_minutes
    derivative = np.gradient(x, 1.0 / fs)
    saccades = _detect_peaks(np.abs(derivative), fs, min_distance=0.2, threshold_factor=4.0)
    saccade_rate = float(saccades.size) / duration_minutes
    if saccades.size > 0:
        peak_v = float(np.median(np.abs(derivative[saccades])))
    else:
        peak_v = 0.0
    return blink_rate, saccade_rate, peak_v


def _eog_eeg_coupling(eog: np.ndarray, eeg: np.ndarray, fs: float) -> Tuple[float, float]:
    """Max lagged correlation and linear regression gain (Eqs. 58–59)."""
    x = eog[0] if eog.ndim == 2 else eog
    y = eeg[0] if eeg.ndim == 2 else eeg
    n = min(x.size, y.size)
    if n < int(fs * 0.5):
        return 0.0, 0.0
    x, y = x[:n], y[:n]
    max_lag = int(0.2 * fs)
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[: n - lag], y[lag:]
        else:
            a, b = x[-lag:], y[: n + lag]
        if a.size < 2:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr) and abs(corr) > abs(best):
            best = float(corr)
    denom = float(np.dot(x, x)) + 1e-8
    gain = float(np.dot(y, x) / denom)
    return _safe_float(best), _safe_float(gain)


def _ecg_self_descriptors(ecg: np.ndarray, fs: float) -> Tuple[float, float, float, np.ndarray]:
    x = ecg[0] if ecg.ndim == 2 else ecg
    r_peaks = _detect_peaks(x, fs, min_distance=0.4, threshold_factor=4.0)
    if r_peaks.size < 2:
        return 0.0, 0.0, 0.0, r_peaks
    rr = np.diff(r_peaks) / fs
    hr = 60.0 / float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size > 1 else 0.0
    return _safe_float(hr), _safe_float(sdnn), _safe_float(rmssd), r_peaks


def _ecg_eeg_coupling(ecg: np.ndarray, eeg: np.ndarray, r_peaks: np.ndarray,
                       fs: float) -> Tuple[float, float]:
    x = ecg[0] if ecg.ndim == 2 else ecg
    y = eeg[0] if eeg.ndim == 2 else eeg
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    hep = 0.0
    if r_peaks.size > 0:
        win = int(0.3 * fs)
        accum = []
        for rp in r_peaks:
            lo, hi = rp + int(0.05 * fs), rp + int(0.05 * fs) + win
            if hi <= y.size:
                accum.append(np.mean(y[lo:hi]))
        if accum:
            hep = float(np.mean(accum))
    phase_eeg = np.angle(hilbert(_bandpass(y, fs, 1.0, 40.0)))
    phase_ecg = np.angle(hilbert(_bandpass(x, fs, 0.5, 40.0)))
    diff = phase_eeg - phase_ecg
    plv = float(np.abs(np.mean(np.exp(1j * diff))))
    return _safe_float(hep), _safe_float(plv)


def _emg_self_descriptors(emg: np.ndarray, fs: float) -> Tuple[float, float, float]:
    x = emg[0] if emg.ndim == 2 else emg
    if x.size < 2:
        return 0.0, 0.0, 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    nperseg = min(x.size, int(fs * 2))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    cumulative = np.cumsum(psd)
    half = cumulative[-1] / 2.0 if cumulative[-1] > 0 else 0.0
    if half > 0:
        idx = int(np.searchsorted(cumulative, half))
        mdf = float(freqs[min(idx, freqs.size - 1)])
    else:
        mdf = 0.0
    probs = psd / (psd.sum() + 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return _safe_float(rms), _safe_float(mdf), _safe_float(entropy)


def _emg_eeg_coupling(emg: np.ndarray, eeg: np.ndarray, fs: float) -> Tuple[float, float]:
    x = emg[0] if emg.ndim == 2 else emg
    y = eeg[0] if eeg.ndim == 2 else eeg
    n = min(x.size, y.size)
    if n < int(fs * 1.0):
        return 0.0, 0.0
    x, y = x[:n], y[:n]
    nperseg = min(n, int(fs * 2))
    freqs, cxy = coherence(x, y, fs=fs, nperseg=nperseg)
    mask = (freqs >= 13.0) & (freqs <= 30.0)
    cmc = float(np.mean(cxy[mask])) if mask.any() else 0.0
    csd = np.fft.rfft(x) * np.conj(np.fft.rfft(y))
    phase = np.unwrap(np.angle(csd))
    rfreqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (rfreqs >= 13.0) & (rfreqs <= 30.0)
    if band.sum() >= 2:
        slope = float(np.polyfit(rfreqs[band], phase[band], 1)[0])
        delay = -slope / (2 * np.pi)
    else:
        delay = 0.0
    return _safe_float(cmc), _safe_float(delay)


def compute_descriptors(eeg: np.ndarray,
                        eog: Optional[np.ndarray] = None,
                        ecg: Optional[np.ndarray] = None,
                        emg: Optional[np.ndarray] = None,
                        fs: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 18-dim descriptor vector and 4-bit modality mask for one window.

    Each modality argument is ``None`` (absent) or an array with shape ``(C, T)`` or
    ``(T,)``. The descriptor vector layout matches the docstring at the top of this file;
    entries belonging to a missing modality are zero.
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim == 1:
        eeg = eeg[None, :]
    delta = np.zeros(NUM_MODALITIES, dtype=np.float32)
    out = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)

    # EOG block (indices 0..4)
    if eog is not None:
        eog = np.asarray(eog, dtype=np.float64)
        if eog.ndim == 1:
            eog = eog[None, :]
        out[0:3] = _eog_self_descriptors(eog, fs)
        out[3:5] = _eog_eeg_coupling(eog, eeg, fs)
        delta[0] = 1.0

    # ECG block (indices 5..9)
    r_peaks: np.ndarray = np.zeros(0, dtype=np.int64)
    if ecg is not None:
        ecg = np.asarray(ecg, dtype=np.float64)
        if ecg.ndim == 1:
            ecg = ecg[None, :]
        hr, sdnn, rmssd, r_peaks = _ecg_self_descriptors(ecg, fs)
        out[5], out[6], out[7] = hr, sdnn, rmssd
        out[8:10] = _ecg_eeg_coupling(ecg, eeg, r_peaks, fs)
        delta[1] = 1.0

    # EMG block (indices 10..14)
    if emg is not None:
        emg = np.asarray(emg, dtype=np.float64)
        if emg.ndim == 1:
            emg = emg[None, :]
        out[10:13] = _emg_self_descriptors(emg, fs)
        out[13:15] = _emg_eeg_coupling(emg, eeg, fs)
        delta[2] = 1.0

    # EEG self block (indices 15..17)
    out[15], out[16], out[17] = _eeg_self_descriptors(eeg, fs)
    delta[3] = 1.0

    return out, delta
