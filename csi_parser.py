"""
csi_parser.py — Parse serial lines into structured CSI frames.

Handles two formats:
  v2:  CSI:<frame>,<timestamp_us>,<num_bytes>:I0,Q0,I1,Q1,...
  v1:  CSI:val0,val1,val2,...            (magnitude-only, legacy)

Produces a CSIFrame with complex CSI, amplitude, and phase for valid subcarriers.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CSIFrame:
    frame_id: int
    timestamp: float            # seconds (host time if firmware ts unavailable)
    complex_csi: np.ndarray     # complex64, valid subcarriers only
    amplitude: np.ndarray       # float64, normalized
    phase: Optional[np.ndarray] # float64, sanitized (None for v1)


class CSIParser:
    def __init__(self, config):
        self.cfg = config
        n_sub = config.total_raw_bytes // 2
        if config.remove_dead:
            self.valid_idx = np.array(
                [i for i in range(n_sub) if i not in set(config.dead_indices)]
            )
        else:
            self.valid_idx = np.arange(n_sub)
        self.n_valid = len(self.valid_idx)

    def parse(self, line: str, host_time: float) -> Optional[CSIFrame]:
        if not line.startswith("CSI:"):
            return None

        payload = line[4:]
        frame_id = -1
        ts = host_time

        # v2 has a header block before a second colon
        if ":" in payload:
            header, _, data_str = payload.partition(":")
            hdr_parts = header.split(",")
            if len(hdr_parts) >= 2:
                try:
                    frame_id = int(hdr_parts[0])
                    # firmware timestamp is microseconds; keep host time for wall-clock
                except ValueError:
                    pass
            raw_str = data_str.strip()
        else:
            raw_str = payload.strip()

        vals = [v for v in raw_str.split(",") if v != ""]
        try:
            raw = np.array([int(v) for v in vals], dtype=np.float64)
        except ValueError:
            return None

        n_expected = self.cfg.total_raw_bytes

        if raw.size == n_expected:
            # I/Q interleaved
            I = raw[0::2]
            Q = raw[1::2]
            csi = (I + 1j * Q).astype(np.complex64)
            csi = csi[self.valid_idx]

            amp = np.abs(csi).astype(np.float64)
            amp = amp / (np.max(amp) + 1e-9)

            phase = np.angle(csi).astype(np.float64)
            phase = self._sanitize_phase(phase)

            return CSIFrame(frame_id, ts, csi, amp, phase)

        elif raw.size == n_expected // 2:
            # magnitude-only legacy
            amp = raw[self.valid_idx]
            amp = amp / (np.max(np.abs(amp)) + 1e-9)
            csi = amp.astype(np.complex64)
            return CSIFrame(frame_id, ts, csi, amp, None)

        return None

    @staticmethod
    def _sanitize_phase(phase: np.ndarray) -> np.ndarray:
        """Remove linear phase slope (sampling time / freq offset)."""
        x = np.arange(phase.size, dtype=np.float64)
        # unwrap first for a stable fit
        unwrapped = np.unwrap(phase)
        a, b = np.polyfit(x, unwrapped, 1)
        corrected = unwrapped - (a * x + b)
        # wrap back to [-pi, pi]
        return np.angle(np.exp(1j * corrected))
