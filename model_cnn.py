"""
model_cnn.py — OPTIONAL deep-learning upgrade (PyTorch).

A small CNN that classifies Doppler spectrograms of CSI motion.
Theoretically the strongest approach, but it needs a LOT more labelled data
than the RandomForest to actually beat it (think thousands of windows per class,
collected across positions/people/environments). Use this only once you have
that dataset. For most setups, model.py (RandomForest) is the right call.

This file is a scaffold — wire it into train.py once you have the data volume.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


if TORCH_OK:

    class CSISpectrogramCNN(nn.Module):
        def __init__(self, n_classes: int, in_ch: int = 1):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))


def csi_window_to_spectrogram(amp_matrix: np.ndarray, fs: float):
    """
    Convert a (W x S) amplitude window to a per-subcarrier Doppler spectrogram
    tensor suitable for the CNN. Returns np.ndarray (S x F x T) or None.
    """
    from scipy.signal import stft
    A = np.asarray(amp_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 32:
        return None
    specs = []
    nper = min(64, A.shape[0])
    for s in range(A.shape[1]):
        sig = A[:, s] - A[:, s].mean()
        _, _, Z = stft(sig, fs=fs, nperseg=nper, noverlap=nper // 2)
        specs.append(np.abs(Z))
    return np.stack(specs, axis=0)
