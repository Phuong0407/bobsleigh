#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SOLVE_RE = re.compile(
    r"Solving for\s+([^,]+),\s+Initial residual\s*=\s*([0-9eE+\-.]+)"
)

TIME_RE = re.compile(r"^\s*Time\s*=\s*([0-9eE+\-.]+)")


def normalize_field(name: str) -> str:
    name = name.strip()

    if name == "p":
        return "Pressure"

    return name


def read_residuals(log_file: Path):
    residuals = {}
    current_iter = -1
    fallback_counter = {}

    with log_file.open("r", errors="ignore") as f:
        for line in f:
            time_match = TIME_RE.match(line)

            if time_match:
                try:
                    current_iter = int(round(float(time_match.group(1))))
                except ValueError:
                    current_iter += 1

            solve_match = SOLVE_RE.search(line)

            if not solve_match:
                continue

            field = normalize_field(solve_match.group(1))
            value = float(solve_match.group(2))

            if current_iter >= 0:
                iteration = current_iter
            else:
                fallback_counter[field] = fallback_counter.get(field, 0) + 1
                iteration = fallback_counter[field]

            if field not in residuals:
                residuals[field] = []

            if residuals[field] and residuals[field][-1][0] == iteration:
                old_iter, old_value = residuals[field][-1]
                residuals[field][-1] = (old_iter, max(old_value, value))
            else:
                residuals[field].append((iteration, value))

    return residuals


def residual_trend(values, window):
    values = np.asarray(values[-window:], dtype=float)

    values = np.maximum(values, 1e-300)
    log_values = np.log10(values)

    x = np.arange(len(log_values))
    slope, _ = np.polyfit(x, log_values, 1)

    decade_change_over_window = slope * window

    return decade_change_over_window


def print_status(residuals, window):
    print()
    print(f"Residual trend over last {window} samples:")

    for field, pairs in residuals.items():
        if len(pairs) < window:
            continue

        values = np.array([v for _, v in pairs], dtype=float)
        trend = residual_trend(values, window)

        if trend < -0.10:
            status = "still decreasing"
        elif abs(trend) <= 0.10:
            status = "plateau / stalled"
        else:
            status = "increasing / suspicious"

        print(
            f"{field:10s}: last={values[-1]:.3e}, "
            f"log10-change/window={trend:+.3f}, "
            f"{status}"
        )


def plot_residuals(residuals, max_iter=None, title="OpenFOAM residuals"):
    plt.figure(figsize=(11, 6))

    preferred_order = ["Ux", "Uy", "Uz", "Pressure", "k", "omega"]

    fields = preferred_order + [
        f for f in residuals.keys() if f not in preferred_order
    ]

    for field in fields:
        if field not in residuals:
            continue

        pairs = residuals[field]
        iterations = np.array([p[0] for p in pairs], dtype=float)
        values = np.array([p[1] for p in pairs], dtype=float)

        if max_iter is not None:
            mask = iterations <= max_iter
            iterations = iterations[mask]
            values = values[mask]

        if len(values) == 0:
            continue

        plt.semilogy(iterations, values, label=field)

    plt.xlabel("Iteration")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("solver")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--save", default=None, help="Save figure, e.g. residuals.png")
    args = parser.parse_args()

#    log_file = Path(f"{args.case}/log.{args.solver}.*")
    case_dir = Path(args.case)
    matches = list(case_dir.glob(f"log.{args.solver}")) + list(
        case_dir.glob(f"log.{args.solver}.*")
    )

    matches = sorted(matches, key=lambda p: p.stat().st_mtime)
    log_file = matches[-1]

    while True:
        if not log_file.exists():
            print(f"Waiting for {log_file}")
            time.sleep(args.sleep)
            continue

        residuals = read_residuals(log_file)

        plt.clf()
        plot_residuals(
            residuals,
            max_iter=args.max_iter,
            title=f"Residual iterative evolution: {log_file}",
        )

        print_status(residuals, args.window)

        if args.save:
            plt.savefig(args.save, dpi=200)

        plt.pause(0.1)

        if not args.watch:
            plt.show()
            break

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
