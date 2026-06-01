#!/usr/bin/env python3

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "case",
    help="OpenFOAM case directory",
)

args = parser.parse_args()

FORCES_DIR = Path(args.case) / "postProcessing" / "forces"

REFRESH_INTERVAL = 0.5
WINDOW = None

plt.ion()

fig, ax = plt.subplots(figsize=(12, 6))

(line_fx,) = ax.plot([], [], label="Fx")
(line_fy,) = ax.plot([], [], label="Fy")
(line_fz,) = ax.plot([], [], label="Fz")

ax.set_xlabel("Time [s]")
ax.set_ylabel("Force [N]")
ax.set_title("OpenFOAM Forces Realtime")

ax.grid(True)
ax.legend()

times = []
fxs = []
fys = []
fzs = []

last_size = 0

def parse_force_line(line):
   line = line.strip()

   if not line or line.startswith("#"):
       return None

   try:
       parts = line.split()

       t = float(parts[0])

       fx = float(parts[1].replace("(", ""))
       fy = float(parts[2])
       fz = float(parts[3].replace(")", ""))

       return t, fx, fy, fz

   except Exception:
       return None

def read_all_force_files(forces_dir):
    data = []

    force_files = sorted(
        forces_dir.glob("*/force.dat"),
        key=lambda p: float(p.parent.name)
    )

    for fp in force_files:
        with open(fp, "r") as f:
            for line in f:
                parsed = parse_force_line(line)
                if parsed is not None:
                    data.append(parsed)

    if not data:
        return np.array([]), np.array([]), np.array([]), np.array([])

    data = np.array(data)

    data = data[np.argsort(data[:, 0])]

    _, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[sorted(unique_idx)]

    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

if not FORCES_DIR.exists():
    raise FileNotFoundError(f"Cannot find: {FORCES_DIR}")

print(f"Monitoring: {FORCES_DIR}")

while True:

    try:
        current_size = sum(fp.stat().st_size for fp in FORCES_DIR.glob("*/force.dat"))

        if current_size != last_size:
            t, fx, fy, fz = read_all_force_files(FORCES_DIR)

            if WINDOW is not None and len(t) > 0:
                mask = t > (t[-1] - WINDOW)
                t, fx, fy, fz = t[mask], fx[mask], fy[mask], fz[mask]

        line_fx.set_data(t, fx)
        line_fy.set_data(t, fy)
        line_fz.set_data(t, fz)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

        last_size = current_size
        time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")
        break

    except Exception as e:
        print("Error:", e)
        time.sleep(1)
