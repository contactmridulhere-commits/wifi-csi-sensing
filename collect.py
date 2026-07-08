"""
collect.py — Collect labelled CSI windows for training.

Usage:
    python collect.py --label walking --seconds 60
    python collect.py --label empty --seconds 60 --synthetic

Records overlapping feature windows tagged with a label into dataset/.
Run once per class. Aim for 60-120 s per class, across a few positions,
to give the RandomForest enough to learn from.
"""
import os
import time
import argparse
import numpy as np

from config import CONFIG
from csi_parser import CSIParser
from features import extract_features, FEATURE_DIM
from sources import make_source


def collect(label: str, seconds: float, synthetic: bool):
    os.makedirs(CONFIG.data_dir, exist_ok=True)
    parser = CSIParser(CONFIG)
    source = make_source(CONFIG, synthetic=synthetic, scenario=label)
    source.start()

    W = CONFIG.feature_window
    amp_buf, phase_buf = [], []
    feats = []

    print(f"[collect] label='{label}' for {seconds}s. Act now...")
    t0 = time.time()
    have_phase = True

    while time.time() - t0 < seconds:
        item = source.read(timeout=2.0)
        if item is None:
            continue
        _, line, host_t = item
        frame = parser.parse(line, host_t)
        if frame is None:
            continue

        amp_buf.append(frame.amplitude)
        if frame.phase is not None:
            phase_buf.append(frame.phase)
        else:
            have_phase = False

        if len(amp_buf) > W:
            amp_buf.pop(0)
            if phase_buf:
                phase_buf.pop(0)

        # Emit a feature vector every W//4 frames (overlapping windows)
        if len(amp_buf) == W and len(amp_buf) % (W // 4) == 0:
            A = np.vstack(amp_buf)
            P = np.vstack(phase_buf) if (have_phase and phase_buf) else None
            fv = extract_features(A, P, CONFIG.sample_rate_hz,
                                  CONFIG.breathing_low_hz, CONFIG.breathing_high_hz)
            feats.append(fv)
            print(f"\r[collect] windows: {len(feats)}", end="", flush=True)

    source.stop()
    feats = np.array(feats)
    if feats.size == 0:
        print("\n[collect] No data captured. Check connection.")
        return

    out = os.path.join(CONFIG.data_dir, f"{label}.npz")
    if os.path.exists(out):
        prev = np.load(out)["X"]
        feats = np.vstack([prev, feats])
    np.savez(out, X=feats, label=label)
    print(f"\n[collect] saved {feats.shape[0]} windows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="activity label")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--synthetic", action="store_true", help="no-hardware test mode")
    args = ap.parse_args()
    collect(args.label, args.seconds, args.synthetic)
