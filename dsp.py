"""
dsp.py — Signal processing primitives for CSI streams.

  - hampel_filter:    robust outlier rejection
  - pca_motion:       SVD-based static/motion separation
  - subcarrier_rank:  pick most motion-sensitive subcarriers
  - BandpassFilter:   zero-phase Butterworth bandpass (breathing / motion bands)
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def hampel_filter(x: np.ndarray, window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Hampel outlier rejection on a 1D signal.
    Replaces points > n_sigmas * (scaled MAD) from the rolling median.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    if n < 2 * window + 1:
        return x
    k = 1.4826  # MAD -> std scaling for normal dist
    for i in range(window, n - window):
        seg = x[i - window:i + window + 1]
        med = np.median(seg)
        mad = k * np.median(np.abs(seg - med))
        if mad > 0 and np.abs(x[i] - med) > n_sigmas * mad:
            x[i] = med
    return x


def pca_motion(csi_amp: np.ndarray, n_components: int = 6, drop_first: bool = True) -> np.ndarray:
    """
    SVD-based motion emphasis on a (time x subcarrier) amplitude matrix.

    The dominant singular component typically captures the static/dominant
    reflection path; motion energy concentrates in the next few components.
    Reconstruct using components [1..n] (optionally dropping component 0)
    to suppress the static background.

    Returns a reconstructed matrix of the same shape.
    """
    M = np.asarray(csi_amp, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] < 3:
        return M
    mean = M.mean(axis=0, keepdims=True)
    Mc = M - mean
    try:
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    except np.linalg.LinAlgError:
        return M
    start = 1 if drop_first else 0
    end = min(n_components, S.size)
    if end <= start:
        return M
    S_sel = np.zeros_like(S)
    S_sel[start:end] = S[start:end]
    recon = (U * S_sel) @ Vt
    return recon + (0.0 if drop_first else mean)


def subcarrier_rank(csi_amp: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the k most motion-sensitive subcarriers,
    ranked by temporal variance over the window.
    """
    M = np.asarray(csi_amp, dtype=np.float64)
    var = M.var(axis=0)
    k = min(k, var.size)
    return np.argsort(var)[::-1][:k]


class BandpassFilter:
    """Zero-phase Butterworth bandpass. Good for breathing / gait bands."""

    def __init__(self, fs: float, low: float, high: float, order: int = 4):
        nyq = fs / 2.0
        low_n = max(1e-4, low / nyq)
        high_n = min(0.999, high / nyq)
        self.sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size < 15:
            return np.zeros_like(x)
        # detrend + normalize for stable filtering
        x = x - np.mean(x)
        s = np.std(x)
        if s < 1e-9:
            return np.zeros_like(x)
        x = x / s
        try:
            return sosfiltfilt(self.sos, x)
        except ValueError:
            return np.zeros_like(x)
