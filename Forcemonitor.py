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

FORCE_FILE = f"{args.case}/postProcessing/forces/0/force.dat"

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


force_path = Path(FORCE_FILE)

if not force_path.exists():
    raise FileNotFoundError(f"Cannot find: {FORCE_FILE}")

print(f"Monitoring: {FORCE_FILE}")

while True:

    try:
        current_size = force_path.stat().st_size

        if current_size != last_size:

            with open(force_path, "r") as f:
                lines = f.readlines()

            times.clear()
            fxs.clear()
            fys.clear()
            fzs.clear()

            for line in lines:
                parsed = parse_force_line(line)

                if parsed is not None:
                    t, fx, fy, fz = parsed

                    times.append(t)
                    fxs.append(fx)
                    fys.append(fy)
                    fzs.append(fz)

            t = np.array(times)
            fx = np.array(fxs)
            fy = np.array(fys)
            fz = np.array(fzs)

            if WINDOW is not None and len(t) > 0:
                mask = t > (t[-1] - WINDOW)

                t = t[mask]
                fx = fx[mask]
                fy = fy[mask]
                fz = fz[mask]

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
