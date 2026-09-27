# v1 prototype

The first version of the CSI sensor (March 2026), kept here for reference. Engine v3 in the repository root replaces it.

- `csi_capture.ino`: ESP32 firmware. Puts the radio in promiscuous mode on channel 1, enables CSI capture and streams each frame's raw I/Q bytes over serial as `CSI:` lines.
- `csi_live_plot.py`: reads the serial stream, drops the dead subcarriers, and plots live subcarrier energy against an adaptive baseline, flagging presence and movement. Set `SERIAL_PORT` at the top of the file.

What v1 could not do, and v3 adds: phase (v1 used amplitude only), denoising, an ML classifier, breathing-band detection and multi-node capture. See the table at the top of the [main README](../README.md).
