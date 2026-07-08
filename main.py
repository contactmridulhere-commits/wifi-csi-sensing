"""
main.py — Real-time CSI sensing pipeline.

Flow per frame:
    serial line -> parse (I/Q -> amp/phase) -> rolling buffer
       -> DSP (Hampel, PCA motion emphasis)
       -> features (fixed-length vector)
       -> detection:
              ML classifier  (if models/activity_rf.joblib exists)
              threshold rules (always-on fallback / bootstrap)
       -> breathing analysis
       -> visualize + log

Run:
    python main.py                 # real hardware (CONFIG.serial_ports)
    python main.py --synthetic     # no hardware, generated signal
    python main.py --scenario present_still --synthetic

Startup performs an empty-room CALIBRATION so the baseline starts sane.
"""
import os
import csv
import time
import argparse
import datetime
import numpy as np

from config import CONFIG
from csi_parser import CSIParser
from dsp import hampel_filter, pca_motion, BandpassFilter
from features import extract_features
from model import ActivityClassifier
from sources import make_source
from visualizer import Dashboard

from collections import deque


def run(synthetic=False, scenario="walking", headless=False):
    for d in [CONFIG.out_dir, CONFIG.log_dir, CONFIG.model_dir]:
        os.makedirs(d, exist_ok=True)

    parser = CSIParser(CONFIG)
    source = make_source(CONFIG, synthetic=synthetic, scenario=scenario)
    source.start()

    # Load ML model if present
    clf = ActivityClassifier.load(CONFIG.model_path) if CONFIG.use_ml else None
    if clf:
        print(f"[ml] loaded classifier: {clf.classes}")
    else:
        print("[ml] no model found — using threshold detector "
              "(run collect.py + train.py to enable ML)")

    breather = BandpassFilter(CONFIG.sample_rate_hz,
                              CONFIG.breathing_low_hz, CONFIG.breathing_high_hz,
                              CONFIG.breathing_filter_order)

    W = CONFIG.feature_window
    amp_buf = deque(maxlen=W)
    phase_buf = deque(maxlen=W)

    disp_amp = deque(maxlen=CONFIG.display_window)
    disp_phase = deque(maxlen=CONFIG.display_window)
    energy_hist = deque(maxlen=CONFIG.plot_window)
    phase_series = deque(maxlen=CONFIG.plot_window)
    baseline_hist = deque(maxlen=CONFIG.plot_window)
    breath_buf = deque(maxlen=512)

    baseline_energy = 1e-6
    presence_on = movement_on = breathing_on = False
    last_alert = 0.0
    frame_count = 0
    have_phase = True

    # CSV log
    csv_file = csv_writer = None
    if CONFIG.log_csv:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CONFIG.log_dir, f"session_{ts_str}.csv")
        csv_file = open(path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time", "energy", "baseline",
                             "ml_label", "ml_conf", "presence", "movement", "breathing"])
        print(f"[log] {path}")

    dash = None
    if not headless:
        dash = Dashboard(CONFIG, parser.n_valid)

    # ── Calibration ──
    print(f"[calib] learning empty room for {CONFIG.calibration_seconds}s — keep area clear...")
    calib_energies = []
    t0 = time.time()
    while time.time() - t0 < CONFIG.calibration_seconds:
        item = source.read(timeout=2.0)
        if item is None:
            continue
        _, line, host_t = item
        fr = parser.parse(line, host_t)
        if fr is None:
            continue
        amp_buf.append(fr.amplitude)
        if len(amp_buf) >= 2:
            calib_energies.append(float(np.abs(fr.amplitude - amp_buf[-2]).sum()))
    if calib_energies:
        baseline_energy = max(1e-6, float(np.median(calib_energies)))
    print(f"[calib] baseline energy = {baseline_energy:.4f}")

    print("[run] live. Ctrl-C to stop.")
    try:
        while True:
            item = source.read(timeout=2.0)
            if item is None:
                continue
            node_id, line, host_t = item
            frame = parser.parse(line, host_t)
            if frame is None:
                continue

            frame_count += 1
            amp_buf.append(frame.amplitude)
            disp_amp.append(frame.amplitude)
            if frame.phase is not None:
                phase_buf.append(frame.phase)
                disp_phase.append(frame.phase)
            else:
                have_phase = False
            breath_buf.append(float(frame.amplitude.mean()))

            # instantaneous energy
            if len(amp_buf) >= 2:
                inst_energy = float(np.abs(amp_buf[-1] - amp_buf[-2]).sum())
            else:
                inst_energy = 0.0
            energy_hist.append(inst_energy)

            # adaptive baseline
            baseline_energy = ((1 - CONFIG.baseline_alpha) * baseline_energy
                               + CONFIG.baseline_alpha * inst_energy)
            baseline_hist.append(baseline_energy)

            ratio = inst_energy / (baseline_energy + 1e-9)

            # phase variability series (for plot)
            if have_phase and len(phase_buf) >= 2:
                d = np.angle(np.exp(1j * (phase_buf[-1] - phase_buf[-2])))
                phase_series.append(float(np.std(d)))
            else:
                phase_series.append(0.0)

            ml_label, ml_conf = "n/a", 0.0
            status, color = "monitoring", "#1a1a2e"

            # ── Detection: run once we have a full feature window ──
            if len(amp_buf) == W and frame_count % 5 == 0:
                A = np.vstack(amp_buf)
                # DSP: outlier-clean the aggregate, PCA motion emphasis
                A_dn = pca_motion(A, CONFIG.pca_components, CONFIG.pca_drop_first)
                P = np.vstack(phase_buf) if (have_phase and len(phase_buf) == W) else None

                fv = extract_features(A_dn, P, CONFIG.sample_rate_hz,
                                      CONFIG.breathing_low_hz, CONFIG.breathing_high_hz)

                if clf is not None:
                    ml_label, ml_conf = clf.predict(fv)

                # threshold fallback
                presence_on = ratio > CONFIG.presence_amp_factor or (
                    have_phase and phase_series[-1] > CONFIG.presence_phase_std)
                movement_on = ratio > CONFIG.movement_amp_factor or (
                    have_phase and phase_series[-1] > CONFIG.movement_phase_std)

                # breathing
                bsig = breather.apply(np.array(breath_buf))
                breath_rms = float(np.sqrt(np.mean(bsig ** 2))) if bsig.size else 0.0
                breathing_on = 0.25 < breath_rms < 1.2 and not movement_on

                # status banner priority: ML > movement > presence > breathing
                now = time.time()
                if clf is not None and ml_conf > 0.55:
                    status = f"{ml_label}  ({ml_conf*100:.0f}%)"
                    color = {"empty": "#2a9d8f", "present_still": "#e9c46a",
                             "walking": "#e63946", "sitting_down": "#f4a261"}.get(ml_label, "#1a1a2e")
                elif movement_on:
                    status, color = "MOVEMENT", "#e63946"
                elif presence_on:
                    status, color = "PRESENCE", "#e9c46a"
                elif breathing_on:
                    status, color = "BREATHING", "#2a9d8f"
                else:
                    status, color = "clear", "#2a9d8f"

                if (movement_on or presence_on) and (now - last_alert) > CONFIG.debounce_seconds:
                    last_alert = now
                    tag = "movement" if movement_on else "presence"
                    print(f"[{tag}] ratio={ratio:.2f} ml={ml_label}({ml_conf:.2f})")
                    if CONFIG.save_on_alert and dash is not None:
                        dash.save(os.path.join(CONFIG.out_dir, f"{tag}_{int(now)}.png"))

            # ── Log ──
            if csv_writer:
                csv_writer.writerow([frame_count, f"{time.time():.3f}",
                                     f"{inst_energy:.4f}", f"{baseline_energy:.4f}",
                                     ml_label, f"{ml_conf:.3f}",
                                     int(presence_on), int(movement_on), int(breathing_on)])

            # ── Visualize ──
            if dash is not None and frame_count % 2 == 0:
                bsig = breather.apply(np.array(breath_buf)) if len(breath_buf) > 30 else None
                dash.update(disp_amp, disp_phase, energy_hist, phase_series,
                            baseline_hist, bsig, status, color)

    except KeyboardInterrupt:
        print("\n[stop] user interrupt")
    finally:
        source.stop()
        if csv_file:
            csv_file.close()
        print("[done] closed cleanly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="no-hardware test mode")
    ap.add_argument("--scenario", default="walking",
                    help="synthetic scenario: empty|present_still|walking|sitting_down")
    ap.add_argument("--headless", action="store_true", help="no GUI (logging only)")
    args = ap.parse_args()
    run(synthetic=args.synthetic, scenario=args.scenario, headless=args.headless)
