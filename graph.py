"""
Generates z-accuracy graphs from accuracy_log.csv.
Shows raw z vs nominal per distance.
Run: python3 graph.py
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from main_2025 import calibrate_z

REPO_DIR = pathlib.Path(__file__).parent
LOG_FILE = REPO_DIR / "accuracy_log.csv"
OUT_DIR = REPO_DIR / "graphs"

NOMINAL_MM = {
    "0.5m": 500,
    "1.0m": 1000,
    "1.5m": 1500,
    "2.0m": 2000,
    "2.5m": 2500,
    "3.0m": 3000,
}

COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple"]

Y_MARGIN_MM = 150  # fixed ±margin around nominal for consistent scale across all panels
Y_LIMITS = {
    "0.5m": (250, 750),  # nominal ±250 — keeps the 500mm nominal line centered
    "2.0m": (1800, 2200),
    "3.0m": (2700, 3300),
}  # per-distance overrides


def plot_distance(ax, dist, raw_z, nominal, color):
    """Plot a single distance panel — usual pipeline calibration applied."""
    z = raw_z.apply(calibrate_z)
    mean_z = z.mean()
    ax.plot(
        z.index, z, color=color, linewidth=0.8, alpha=0.85, label=f"measured z (μ={mean_z:.0f}mm)"
    )
    if nominal:
        ax.axhline(
            nominal, color="red", linestyle="--", linewidth=1.2, label=f"nominal ({nominal}mm)"
        )
    ax.axhline(mean_z, color="gray", linestyle=":", linewidth=1.2, label=f"mean ({mean_z:.0f}mm)")
    ymin, ymax = Y_LIMITS.get(dist, (nominal - Y_MARGIN_MM, nominal + Y_MARGIN_MM))
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Distance: {dist} (bad lighting)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("z (mm)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main() -> None:
    """Read CSV and produce combined + individual PNGs per distance."""
    df = pd.read_csv(LOG_FILE)
    OUT_DIR.mkdir(exist_ok=True)

    distances = sorted(df["test_distance"].unique())
    n = len(distances)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("OAK-D SpatialDetectionNetwork — z accuracy over time (bad lighting)", fontsize=13)

    for ax, dist, color in zip(axes, distances, COLORS):
        subset = df[df["test_distance"] == dist].copy().sort_values("timestamp")
        raw_z = subset["z_mm"].reset_index(drop=True)
        nominal = NOMINAL_MM.get(dist)
        plot_distance(ax, dist, raw_z, nominal, color)

    fig.tight_layout()
    out_path = OUT_DIR / "z_accuracy_all_bad_lighting.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    for dist, color in zip(distances, COLORS):
        subset = df[df["test_distance"] == dist].copy().sort_values("timestamp")
        raw_z = subset["z_mm"].reset_index(drop=True)
        nominal = NOMINAL_MM.get(dist)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        plot_distance(ax2, dist, raw_z, nominal, color)
        fig2.tight_layout()
        fname = OUT_DIR / f"z_{dist.replace('.', '_')}_bad_lighting.png"
        fig2.savefig(fname, dpi=150)
        plt.close(fig2)
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
