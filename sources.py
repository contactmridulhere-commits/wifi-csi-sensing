"""
sources.py — CSI input abstraction.

SingleNodeSource:  one ESP32 on one serial port.
MultiNodeSource:   several ESP32s (one TX beacon + N RX), fused frame-by-frame.

Multi-node is the real path to spatial awareness: each RX node sees the target
from a different angle, so fusing their motion features gives crude zone-level
localization. Add ports to CONFIG.serial_ports to enable it.
"""
import time
import threading
import queue
import numpy as np

try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False


class _PortReader(threading.Thread):
    """Reads one serial port in a background thread into a queue."""
    def __init__(self, port, baud, out_q, node_id):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_q = out_q
        self.node_id = node_id
        self._stop = threading.Event()
        self.ser = None

    def _connect(self):
        while not self._stop.is_set():
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=1)
                print(f"[serial] node {self.node_id} connected on {self.port}")
                return
            except Exception:
                print(f"[serial] waiting for {self.port} (node {self.node_id})...")
                time.sleep(2)

    def run(self):
        if not SERIAL_OK:
            print("[serial] pyserial not installed — running in NO-HARDWARE mode.")
            return
        self._connect()
        while not self._stop.is_set():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line.startswith("CSI:"):
                    self.out_q.put((self.node_id, line, time.time()))
            except Exception:
                print(f"[serial] node {self.node_id} lost — reconnecting")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self._connect()

    def stop(self):
        self._stop.set()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


class MultiNodeSource:
    def __init__(self, ports, baud):
        self.q = queue.Queue(maxsize=2000)
        self.readers = [
            _PortReader(p, baud, self.q, node_id=i) for i, p in enumerate(ports)
        ]
        self.n_nodes = len(ports)

    def start(self):
        for r in self.readers:
            r.start()

    def read(self, timeout=1.0):
        """Yield (node_id, line, host_time) as they arrive."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        for r in self.readers:
            r.stop()


class SyntheticSource:
    """
    NO-HARDWARE test source. Generates plausible CSI lines so you can develop
    and validate the full pipeline without an ESP32 plugged in.
    """
    def __init__(self, config, scenario="walking"):
        self.cfg = config
        self.scenario = scenario
        self.n = config.total_raw_bytes
        self.t = 0
        self.n_nodes = 1

    def start(self):
        pass

    def read(self, timeout=1.0):
        time.sleep(1.0 / self.cfg.sample_rate_hz)
        self.t += 1
        base = np.random.randn(self.n) * 3
        if self.scenario in ("walking", "sitting_down"):
            amp = 30 * np.sin(2 * np.pi * 1.5 * self.t / self.cfg.sample_rate_hz)
            base += amp
        elif self.scenario == "present_still":
            amp = 6 * np.sin(2 * np.pi * 0.3 * self.t / self.cfg.sample_rate_hz)  # breathing
            base += amp
        vals = np.clip(base + 64, -128, 127).astype(int)
        line = f"CSI:{self.t},{self.t*20000},{self.n}:" + ",".join(map(str, vals))
        return (0, line, time.time())

    def stop(self):
        pass


def make_source(config, synthetic=False, scenario="walking"):
    if synthetic or not SERIAL_OK:
        return SyntheticSource(config, scenario)
    return MultiNodeSource(config.serial_ports, config.baud)
