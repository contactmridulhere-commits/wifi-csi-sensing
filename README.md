# WiFi CSI Sensing · Engine v3

A modular WiFi Channel State Information (CSI) sensing system for the ESP32.
Detects presence, motion, and micro-motion (breathing-band) in an environment
by analyzing how WiFi signals are perturbed by bodies in the space.
No camera, no wearable.

**Build video:** [WiFi sensing on an ESP32](https://youtu.be/Qr2lyrrXwPI)

This is a full rebuild of the original two-file prototype, which is kept in
[`v1-prototype`](v1-prototype), into a proper signal-processing + machine-learning pipeline.

---

## What changed from the prototype

| Prototype (v1)                | This build (v3)                                        |
|-------------------------------|--------------------------------------------------------|
| Amplitude only                | Full I/Q → amplitude **and** phase                     |
| Raw phase                     | Phase sanitized (linear STO/SFO removal)               |
| Single hand-tuned threshold   | Adaptive threshold **+** ML classifier                 |
| No denoising                  | Hampel outlier rejection + PCA static/motion separation |
| No motion typing              | RandomForest activity classification                   |
| No vitals                     | Breathing-band bandpass + autocorrelation periodicity  |
| Single script                 | Modular package (parse / dsp / features / model / viz) |
| Single ESP32 only             | Multi-node architecture (TX beacon + N RX)             |
| Passive capture               | Active null-frame injection (reliable CSI)             |
| No data tooling               | Labelled collection + training scripts                 |

---

## Architecture

```
        ESP32(s)  ──serial──▶  sources.py ──▶ csi_parser.py ──▶ rolling buffer
                                                                      │
                                          dsp.py (Hampel, PCA) ◀──────┤
                                                    │                 │
                                          features.py (16-D vector)   │
                                                    │                 │
                          ┌─────────────────────────┼─────────────────┐
                          ▼                         ▼                 ▼
                  model.py (RandomForest)   threshold detector   breathing band
                          │                         │                 │
                          └────────────▶ main.py orchestrator ◀───────┘
                                              │
                                  visualizer.py + CSV log
```

### Files
- `config.py` — all parameters (dataclass)
- `csi_parser.py` — serial → structured `CSIFrame`, I/Q decomposition, phase sanitization
- `dsp.py` — Hampel filter, PCA motion emphasis, subcarrier ranking, Butterworth bandpass
- `features.py` — fixed-length 16-D feature vector (amplitude, motion, phase, Doppler, periodicity)
- `model.py` — RandomForest activity classifier (train/save/load/predict)
- `model_cnn.py` — **optional** PyTorch CNN scaffold (needs much more data)
- `sources.py` — single-node, multi-node, and synthetic (no-hardware) sources
- `collect.py` — record labelled windows for training
- `train.py` — train + cross-validate + save the classifier
- `visualizer.py` — live light-themed 4-panel dashboard
- `main.py` — real-time pipeline
- `firmware_v3.ino` — upgraded ESP32 firmware

---

## Setup

```bash
pip install numpy scipy scikit-learn joblib matplotlib pyserial
# optional deep-learning path:
# pip install torch
```

Flash `firmware_v3.ino` to the ESP32 (Arduino IDE, ESP32 board package).
Set `SERIAL_PORT` list in `config.py` to match your device (e.g. `COM5`).

---

## Usage

**Try it with no hardware first** (synthetic signal, proves the pipeline):
```bash
python main.py --synthetic --scenario walking
```

**Run live:**
```bash
python main.py
```
Startup runs an 8-second empty-room calibration — keep the area clear.

**Enable the ML classifier** (recommended — big accuracy gain over thresholds):
```bash
# 1. collect ~60-120 s per class, acting out each one
python collect.py --label empty         --seconds 90
python collect.py --label present_still  --seconds 90
python collect.py --label walking        --seconds 90
python collect.py --label sitting_down   --seconds 90

# 2. train
python train.py

# 3. run — main.py auto-loads the model
python main.py
```

---

## Multi-node (the real path to localization)

A single ESP32 gives you one spatial view: it can tell you *something* changed,
not *where*. To get zone-level localization:

1. Flash one board with `ROLE_RX 0` (TX beacon).
2. Flash 2-3 boards with `ROLE_RX 1`, placed around the area at different angles.
3. Add each RX serial port to `CONFIG.serial_ports`.
4. `MultiNodeSource` fuses them; extend the feature stage to concatenate
   per-node vectors for a spatial classifier.

---

## Honest capability ceiling

This extracts essentially everything a commodity ESP32 CSI setup can physically
produce. It will **not**:

- image through walls or reconstruct a scene,
- identify *who* someone is,
- give precise (sub-metre) localization from a single node,
- provide clinical-grade vital signs.

Those require more antennas (MIMO arrays), synchronized clocks, or purpose-built
FMCW radar (e.g. TI IWR / Infineon BGT60) — a hardware step, not a code step.
The physics, not the software, is the limit. What this build *does* is turn a
noisy side-channel into a robust, ML-backed presence/motion/breathing sensor,
with a clean architecture ready to scale when you add nodes.

Legitimate uses: occupancy sensing, smart-home automation, elderly fall/inactivity
monitoring, presence-aware energy control, and research/education. Deploy it on
spaces you own or have permission to monitor.

---

## License

MIT. See [LICENSE](LICENSE).
