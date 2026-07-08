"""
visualizer.py — Live light-themed dashboard.

Four panels:
  1. Amplitude heatmap   (subcarrier x time)
  2. Phase heatmap       (sanitized)
  3. Motion + baselines  (amplitude energy & phase variance)
  4. Breathing band      (filtered micro-motion)

Plus a status banner showing the current classification / detection.
Light background, bold typography — clean and readable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Dashboard:
    def __init__(self, config, n_valid):
        self.cfg = config
        self.W = config.display_window
        self.S = n_valid

        plt.ion()
        plt.style.use("default")  # light background per preference
        self.fig = plt.figure(figsize=(14, 9), facecolor="white")
        self.fig.suptitle("CSI Sensing Engine  ·  live",
                          fontsize=15, fontweight="bold", color="#1a1a2e")
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.4, wspace=0.22)

        # Amplitude heatmap
        self.ax_amp = self.fig.add_subplot(gs[0, 0])
        self.amp_img = self.ax_amp.imshow(
            np.zeros((self.W, self.S)), aspect="auto", vmin=0, vmax=1,
            interpolation="nearest", cmap="viridis")
        self.ax_amp.set_title("Amplitude", fontsize=11, fontweight="bold")
        self.ax_amp.set_ylabel("time")
        self.ax_amp.set_xlabel("subcarrier")

        # Phase heatmap
        self.ax_phase = self.fig.add_subplot(gs[0, 1])
        self.phase_img = self.ax_phase.imshow(
            np.zeros((self.W, self.S)), aspect="auto", vmin=-np.pi, vmax=np.pi,
            interpolation="nearest", cmap="RdBu")
        self.ax_phase.set_title("Phase (sanitized)", fontsize=11, fontweight="bold")
        self.ax_phase.set_ylabel("time")
        self.ax_phase.set_xlabel("subcarrier")

        # Motion energy
        self.ax_e = self.fig.add_subplot(gs[1, :])
        (self.l_amp,) = self.ax_e.plot([], [], lw=1.4, color="#e63946", label="amp energy")
        (self.l_phase,) = self.ax_e.plot([], [], lw=1.2, color="#1d3557", label="phase var (x10)")
        (self.l_bamp,) = self.ax_e.plot([], [], lw=1, ls="--", color="#e6a1a1", label="amp baseline")
        self.ax_e.set_title("Motion Energy", fontsize=11, fontweight="bold")
        self.ax_e.set_ylabel("energy"); self.ax_e.set_xlabel("frame")
        self.ax_e.legend(loc="upper right", fontsize=8)
        self.ax_e.grid(alpha=0.2)

        # Breathing band
        self.ax_b = self.fig.add_subplot(gs[2, :])
        (self.l_breath,) = self.ax_b.plot([], [], lw=1.3, color="#2a9d8f",
                                          label=f"breathing band "
                                                f"({config.breathing_low_hz}-{config.breathing_high_hz} Hz)")
        self.ax_b.set_title("Breathing / Micro-Motion", fontsize=11, fontweight="bold")
        self.ax_b.set_ylabel("filtered"); self.ax_b.set_xlabel("sample")
        self.ax_b.legend(loc="upper right", fontsize=8)
        self.ax_b.grid(alpha=0.2)

        # Status banner
        self.banner = self.ax_amp.text(
            0.02, 1.28, "starting…", transform=self.ax_amp.transAxes,
            fontsize=13, fontweight="bold", color="#1a1a2e",
            bbox=dict(facecolor="#f1faee", edgecolor="#457b9d", boxstyle="round,pad=0.4"))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(self, amp_hist, phase_hist, energy_hist, phase_hist_series,
               baseline_amp_hist, breathing_signal, status, color="#1a1a2e"):
        # amplitude heatmap
        if len(amp_hist) > 0:
            mat = np.vstack(list(amp_hist)[-self.W:])
            if mat.shape[0] < self.W:
                mat = np.vstack([np.zeros((self.W - mat.shape[0], self.S)), mat])
            self.amp_img.set_data(mat)
            self.amp_img.set_clim(0, max(0.01, mat.max()))

        # phase heatmap
        if len(phase_hist) > 0:
            matp = np.vstack(list(phase_hist)[-self.W:])
            if matp.shape[0] < self.W:
                matp = np.vstack([np.zeros((self.W - matp.shape[0], self.S)), matp])
            self.phase_img.set_data(matp)

        # energy plot
        e = np.array(energy_hist)
        p = np.array(phase_hist_series)
        b = np.array(baseline_amp_hist)
        self.l_amp.set_data(np.arange(e.size), e)
        self.l_phase.set_data(np.arange(p.size), p * 10)
        self.l_bamp.set_data(np.arange(b.size), b)
        if e.size > 0:
            self.ax_e.set_xlim(0, max(200, e.size))
            self.ax_e.set_ylim(0, max(1.0, e.max() * 1.2))

        # breathing plot
        if breathing_signal is not None and breathing_signal.size > 5:
            self.l_breath.set_data(np.arange(breathing_signal.size), breathing_signal)
            self.ax_b.set_xlim(0, breathing_signal.size)
            self.ax_b.set_ylim(-3, 3)

        self.banner.set_text(status)
        self.banner.set_color(color)
        plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
