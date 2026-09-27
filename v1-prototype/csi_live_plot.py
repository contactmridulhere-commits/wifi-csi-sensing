import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from scipy.ndimage import gaussian_filter1d

SERIAL_PORT = "COM5"     
BAUD = 115200
TOTAL_SUBCARRIERS = 128
WINDOW = 60               
REMOVE_DEAD = True       
SAVE_ON_ALERT = True     
OUT_DIR = "captures"      

ENERGY_SMOOTH = 2.0      
BASELINE_ALPHA = 0.02   
PRESENCE_FACTOR = 3.0   
MOVEMENT_FACTOR = 6.0    
DEBOUNCE_SECONDS = 1.0   


DEAD_INDICES = set(list(range(0,6)) + list(range(56,65)) + list(range(122,128)))

if SAVE_ON_ALERT and not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if REMOVE_DEAD:
    VALID_IDX = [i for i in range(TOTAL_SUBCARRIERS) if i not in DEAD_INDICES]
else:
    VALID_IDX = list(range(TOTAL_SUBCARRIERS))
CLEAN = len(VALID_IDX)

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)


history = deque(maxlen=WINDOW)            
energy_history = deque(maxlen=WINDOW*4)  

baseline_energy = 1e-6
last_alert_time = 0.0
presence_on = False
movement_on = False


plt.ion()
fig = plt.figure(figsize=(12,7))
ax_heat = fig.add_subplot(2,1,1)
ax_energy = fig.add_subplot(2,1,2)

heat_img = ax_heat.imshow(np.zeros((WINDOW, CLEAN)), aspect='auto', vmin=-1, vmax=1,
                          interpolation='nearest', cmap='viridis')
ax_heat.set_title("CSI Heatmap (recent frames)")
ax_heat.set_ylabel("Time")

energy_line, = ax_energy.plot([], [], lw=1)
ax_energy.set_title("Motion Energy (normalized)")
ax_energy.set_ylabel("Energy")
ax_energy.set_xlabel("Time (frames)")

status_text = ax_heat.text(0.01, 0.95, "", transform=ax_heat.transAxes, color='yellow',
                           fontsize=12, bbox=dict(facecolor='black', alpha=0.6))

plt.tight_layout()

def parse_csi_line(line):
    """Expecting lines like: CSI:-128,1,24,0,-12,... (TOTAL values)"""
    if not line.startswith("CSI:"):
        return None
    payload = line.split("CSI:", 1)[1].strip()
    parts = [p.strip() for p in payload.split(",") if p.strip() != ""]
    if len(parts) != TOTAL_SUBCARRIERS:
        return None
    try:
        arr = np.array([int(x) for x in parts], dtype=np.float32)
        
        cleaned = arr[VALID_IDX]
    
        cleaned = cleaned / 60.0
        return cleaned
    except:
        return None

def compute_energy(curr, prev):
    """Energy as sum of absolute difference across subcarriers"""
    if prev is None:
        return 0.0
    diff = np.abs(curr - prev)
    return float(np.sum(diff))

def update_baseline(energy):
    global baseline_energy
    baseline_energy = (1 - BASELINE_ALPHA) * baseline_energy + BASELINE_ALPHA * energy
    return baseline_energy

def save_snapshot():
    ts = int(time.time())
    fname = os.path.join(OUT_DIR, f"alert_{ts}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"[saved] {fname}")

print("Waiting for CSI lines.")

try:
    while True:
        raw = ser.readline().decode(errors='ignore').strip()
        csi = parse_csi_line(raw)
        if csi is None:
            continue

        history.append(csi)
        prev = history[-2] if len(history) >= 2 else None

        energy = compute_energy(csi, prev)
        energy_history.append(energy)

        if len(energy_history) >= 5:
            smooth_en = gaussian_filter1d(np.array(energy_history), sigma=ENERGY_SMOOTH)
            recent_energy = float(smooth_en[-1])
        else:
            recent_energy = energy

        b = update_baseline(recent_energy)

        now = time.time()
        presence_rule = (b > 0 and recent_energy / (b + 1e-9) > PRESENCE_FACTOR)
        movement_rule = (b > 0 and recent_energy / (b + 1e-9) > MOVEMENT_FACTOR)

        if presence_rule and (now - last_alert_time) > DEBOUNCE_SECONDS:
            if not presence_on:
                print("HUMAN PRESENCE DETECTED (energy {:.2f}, baseline {:.2f})".format(recent_energy, b))
                presence_on = True
                last_alert_time = now
                if SAVE_ON_ALERT:
                    save_snapshot()
        
        if presence_on and recent_energy / (b + 1e-9) < (PRESENCE_FACTOR * 0.7):
            presence_on = False

        if movement_rule and (now - last_alert_time) > DEBOUNCE_SECONDS:
            if not movement_on:
                print("MOVEMENT THROUGH WALL DETECTED (energy {:.2f}, baseline {:.2f})".format(recent_energy, b))
                movement_on = True
                last_alert_time = now
                if SAVE_ON_ALERT:
                    save_snapshot()
        if movement_on and recent_energy / (b + 1e-9) < (MOVEMENT_FACTOR * 0.7):
            movement_on = False

        mat = np.vstack(history) if len(history) > 0 else np.zeros((WINDOW, CLEAN))
        if mat.shape[0] < WINDOW:
            pad = np.zeros((WINDOW - mat.shape[0], CLEAN))
            mat = np.vstack([pad, mat])

        heat_img.set_data(mat)
        heat_img.set_clim(vmin=np.min(mat), vmax=np.max(mat))

        en_array = np.array(energy_history)
        energy_line.set_data(np.arange(len(en_array)), en_array)
        ax_energy.set_xlim(0, max(200, len(en_array)))
        ax_energy.set_ylim(0, max(1.0, np.max(en_array) * 1.2))

        status = []
        if presence_on: status.append("PRESENCE")
        if movement_on: status.append("MOVEMENT")
        status_txt = " | ".join(status) if status else "No presence"
        status_text.set_text(status_txt)

        plt.pause(0.001)

except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print("Error:", e)
finally:
    ser.close()