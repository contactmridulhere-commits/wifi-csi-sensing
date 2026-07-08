"""
features.py — Extract a fixed-length feature vector from a window of CSI frames.

The vector length is independent of subcarrier count (everything is aggregated
across subcarriers), so the same model works regardless of dead-subcarrier config.

Feature groups:
  Amplitude statistics ..... overall level & spread of motion
  Motion energy ............ frame-to-frame amplitude change
  Phase dynamics ........... circular phase variability
  Doppler band energies .... STFT of aggregate motion -> velocity bands
  Periodicity .............. autocorrelation peak in breathing band
  Spatial spread ........... fraction of active subcarriers
"""
import numpy as np
from scipy.signal import stft


FEATURE_NAMES = [
    "amp_mean", "amp_std", "amp_max", "amp_range",
    "motion_mean", "motion_std", "motion_max",
    "phase_std_mean", "phase_std_max",
    "dop_static", "dop_slow", "dop_med", "dop_fast",
    "acf_peak", "acf_lag_norm",
    "active_frac",
]
FEATURE_DIM = len(FEATURE_NAMES)


def _doppler_bands(motion_signal: np.ndarray, fs: float) -> np.ndarray:
    """Energy in 4 velocity bands via STFT of the aggregate motion signal."""
    if motion_signal.size < 32:
        return np.zeros(4)
    nper = min(64, motion_signal.size)
    try:
        f, _, Z = stft(motion_signal, fs=fs, nperseg=nper, noverlap=nper // 2)
    except ValueError:
        return np.zeros(4)
    P = np.mean(np.abs(Z) ** 2, axis=1)  # avg power spectrum
    total = np.sum(P) + 1e-12
    bands = [(0.0, 0.5), (0.5, 2.0), (2.0, 8.0), (8.0, fs / 2)]
    out = []
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        out.append(float(np.sum(P[mask]) / total))
    return np.array(out)


def _autocorr_periodicity(signal: np.ndarray, fs: float,
                          low_hz: float, high_hz: float):
    """Peak autocorrelation within the breathing band -> (peak, normalized lag)."""
    if signal.size < 32:
        return 0.0, 0.0
    x = signal - np.mean(signal)
    s = np.std(x)
    if s < 1e-9:
        return 0.0, 0.0
    x = x / s
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]
    ac = ac / (ac[0] + 1e-12)
    lag_lo = max(1, int(fs / high_hz))
    lag_hi = min(ac.size - 1, int(fs / low_hz))
    if lag_hi <= lag_lo:
        return 0.0, 0.0
    seg = ac[lag_lo:lag_hi]
    peak = float(np.max(seg))
    lag = float(np.argmax(seg) + lag_lo)
    return peak, lag / (ac.size + 1e-9)


def extract_features(amp_matrix: np.ndarray,
                     phase_matrix,
                     fs: float,
                     breathing_low: float = 0.15,
                     breathing_high: float = 0.6) -> np.ndarray:
    """
    amp_matrix:   (W x S) normalized amplitude
    phase_matrix: (W x S) sanitized phase, or None
    Returns:      1D float array of length FEATURE_DIM
    """
    A = np.asarray(amp_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 3:
        return np.zeros(FEATURE_DIM)

    # Amplitude statistics
    amp_mean = float(A.mean())
    amp_std = float(A.std())
    amp_max = float(A.max())
    amp_range = float(A.max() - A.min())

    # Motion energy: frame-to-frame L1 change, summed across subcarriers
    diffs = np.abs(np.diff(A, axis=0)).sum(axis=1)
    motion_mean = float(diffs.mean())
    motion_std = float(diffs.std())
    motion_max = float(diffs.max())

    # Phase dynamics
    if phase_matrix is not None:
        P = np.asarray(phase_matrix, dtype=np.float64)
        # circular std across subcarriers, per frame
        per_frame_std = []
        for row in P:
            c = np.abs(np.mean(np.exp(1j * row)))
            per_frame_std.append(np.sqrt(-2.0 * np.log(max(c, 1e-9))))
        per_frame_std = np.array(per_frame_std)
        phase_std_mean = float(per_frame_std.mean())
        phase_std_max = float(per_frame_std.max())
    else:
        phase_std_mean = 0.0
        phase_std_max = 0.0

    # Doppler bands on aggregate motion signal (mean amplitude over subcarriers)
    agg = A.mean(axis=1)
    dop = _doppler_bands(agg - agg.mean(), fs)

    # Periodicity (breathing) via autocorrelation
    acf_peak, acf_lag = _autocorr_periodicity(agg, fs, breathing_low, breathing_high)

    # Spatial spread: fraction of subcarriers whose variance exceeds median variance
    var = A.var(axis=0)
    active_frac = float(np.mean(var > (np.median(var) + 1e-9)))

    return np.array([
        amp_mean, amp_std, amp_max, amp_range,
        motion_mean, motion_std, motion_max,
        phase_std_mean, phase_std_max,
        dop[0], dop[1], dop[2], dop[3],
        acf_peak, acf_lag,
        active_frac,
    ], dtype=np.float64)
