"""
config.py — Central configuration for the CSI sensing engine.
All tunable parameters live here. Import CONFIG everywhere.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Serial / hardware ──
    serial_ports: List[str] = field(default_factory=lambda: ["COM5"])  # add more for multi-node
    baud: int = 115200
    total_raw_bytes: int = 128          # bytes from ESP32 (I/Q interleaved)

    # ── Subcarrier layout ──
    # 64 complex subcarriers after I/Q decomposition.
    # Guard bands + DC null removed.
    remove_dead: bool = True
    dead_indices: List[int] = field(
        default_factory=lambda: list(range(0, 3)) + list(range(28, 37)) + list(range(60, 64))
    )

    # ── Timing ──
    sample_rate_hz: float = 50.0        # approx CSI frame rate (matches firmware injection)

    # ── Windows ──
    feature_window: int = 128           # frames per feature vector (~2.5 s @ 50 Hz)
    display_window: int = 120           # frames shown in heatmaps
    plot_window: int = 400              # frames in the time-series plots

    # ── DSP ──
    hampel_window: int = 5
    hampel_sigmas: float = 3.0
    pca_components: int = 6              # components kept for motion emphasis
    pca_drop_first: bool = True         # drop dominant static component

    # ── Breathing band ──
    breathing_low_hz: float = 0.15      # ~9 bpm
    breathing_high_hz: float = 0.6      # ~36 bpm
    breathing_filter_order: int = 4

    # ── Threshold detector (fallback / bootstrap) ──
    baseline_alpha: float = 0.015
    energy_smooth_sigma: float = 2.0
    presence_amp_factor: float = 2.5
    movement_amp_factor: float = 5.0
    presence_phase_std: float = 0.15
    movement_phase_std: float = 0.40
    debounce_seconds: float = 1.5

    # ── ML ──
    model_path: str = "models/activity_rf.joblib"
    use_ml: bool = True                 # if a trained model exists, use it
    ml_classes: List[str] = field(
        default_factory=lambda: ["empty", "present_still", "walking", "sitting_down"]
    )

    # ── I/O ──
    out_dir: str = "captures"
    log_dir: str = "logs"
    data_dir: str = "dataset"           # labelled windows for training
    model_dir: str = "models"
    save_on_alert: bool = True
    log_csv: bool = True

    # ── Calibration ──
    calibration_seconds: float = 8.0    # empty-room learning at startup


CONFIG = Config()
