#!/usr/bin/env python3
# Usage:
#   python3 visualize_costs.py data/p0.04_hist.txt
#   python3 visualize_costs.py data/p0.04_hist.txt --out data/p0.04_hist.png --title "p=0.04 cost distribution"
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_costs(path: Path):
    xs = []
    ps = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            xs.append(int(parts[0]))
            ps.append(float(parts[1]))
    return xs, ps


def compute_stats(xs, ps):
    mu = sum(x * p for x, p in zip(xs, ps))
    m2 = sum((x * x) * p for x, p in zip(xs, ps))
    var = m2 - mu * mu
    return mu, var


def plot_dist(xs, ps, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ps, width=0.8, color="#4C78A8")
    ax.set_xlabel("iterations")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main():
    ap = argparse.ArgumentParser(description="Visualize iteration probability distribution from iteration histogram file.")
    ap.add_argument("input", help="histogram file (e.g., data0/p0.04_hist.txt)")
    ap.add_argument("--out", default="", help="output PNG path (default: <input>.png)")
    ap.add_argument("--title", default="", help="plot title")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    xs, ps = read_costs(in_path)
    if not xs:
        raise SystemExit("No data rows found.")

    mu, var = compute_stats(xs, ps)
    title = args.title or f"{in_path.name} (mean={mu:.3f}, var={var:.3f})"
    out_png = Path(args.out) if args.out else in_path.with_suffix(in_path.suffix + ".png")

    plot_dist(xs, ps, title, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
