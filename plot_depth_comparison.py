"""Plot RSA/encoding score comparisons between depth 1 and depth 8
(transitions1, untied norm, expansion8, 50ep) across epochs 5/10/25/50.

Produces:
  - depth1_progression.png, depth8_progression.png: line charts of score vs.
    epoch, one panel per analysis (RSA / encoding), one line per region.
  - epoch{5,10,25,50}_comparison.png: grouped bar charts comparing depth 1 vs
    depth 8 for each (region, analysis) combination at a fixed epoch.

Usage:
    python plot_depth_comparison.py
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_RESULTS_DB_PATH = Path("results.db")

_CHECKPOINT_DIRS = {
    "depth1": "model_checkpoints/ectiednet_imagenet1k_depth1_transitions1_untiednorm_expansion8_50ep",
    "depth8": "model_checkpoints/ectiednet_imagenet1k_depth8_transitions1_untiednorm_expansion8_50ep",
}
_EPOCHS = [5, 10, 25, 50]
_ANALYSES = ["rsa", "encoding_score"]
_ANALYSIS_LABELS = {"rsa": "RSA", "encoding_score": "Encoding"}
_REGIONS = ["early visual stream", "ventral visual stream"]
_REGION_LABELS = {"early visual stream": "Early", "ventral visual stream": "Ventral"}

# validated categorical slots 1 & 2 (blue / orange) -- see dataviz skill palette.md
_REGION_COLORS = {"early visual stream": "#2a78d6", "ventral visual stream": "#eb6834"}
_DEPTH_COLORS = {"depth1": "#2a78d6", "depth8": "#eb6834"}


def load_scores() -> pd.DataFrame:
    conn = sqlite3.connect(str(_RESULTS_DB_PATH))
    placeholders = ",".join("?" * len(_CHECKPOINT_DIRS))
    df = pd.read_sql_query(
        f"SELECT checkpoint_dir, analysis, region, subject_idx, epoch, score "
        f"FROM results WHERE checkpoint_dir IN ({placeholders})",
        conn, params=list(_CHECKPOINT_DIRS.values()),
    )
    conn.close()
    if df.empty:
        raise ValueError(f"No rows in {_RESULTS_DB_PATH} for the depth1/depth8 checkpoint dirs")

    dir_to_depth = {v: k for k, v in _CHECKPOINT_DIRS.items()}
    df["depth"] = df["checkpoint_dir"].map(dir_to_depth)

    # mean across subjects -- matches the "Mean" line evals.py prints per (region, analysis, epoch)
    return df.groupby(["depth", "analysis", "region", "epoch"])["score"].mean().reset_index()


def _style_axis(ax):
    ax.grid(alpha=0.3, color="#e1e0d9")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")


def plot_progression(df: pd.DataFrame, depth: str, out_path: str):
    """Line chart: score vs. epoch, one panel per analysis, one line per region."""
    sub = df[df["depth"] == depth]
    missing = sorted(set(_EPOCHS) - set(sub["epoch"].unique()))
    if missing:
        print(f"  [warn] {depth}: no data for epoch(s) {missing} -- run those evals first")

    fig, axes = plt.subplots(1, len(_ANALYSES), figsize=(6 * len(_ANALYSES), 4.5))
    for ax, analysis in zip(axes, _ANALYSES):
        for region in _REGIONS:
            line = sub[(sub["analysis"] == analysis) & (sub["region"] == region)].sort_values("epoch")
            if line.empty:
                continue
            ax.plot(
                line["epoch"], line["score"],
                marker="o", markersize=6, linewidth=2,
                color=_REGION_COLORS[region], label=_REGION_LABELS[region],
            )
        ax.set_title(_ANALYSIS_LABELS[analysis])
        ax.set_xlabel("epoch")
        ax.set_ylabel("score (mean across subjects)")
        ax.set_xticks(_EPOCHS)
        ax.legend(frameon=False)
        _style_axis(ax)

    fig.suptitle(f"{depth}: score progression over training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_epoch_comparison(df: pd.DataFrame, epoch: int, out_path: str):
    """Grouped bar chart: depth1 vs depth8, grouped by (region, analysis)."""
    sub = df[df["epoch"] == epoch]
    if sub.empty:
        print(f"  [warn] epoch {epoch}: no data -- skipping {out_path}")
        return

    groups = [(region, analysis) for region in _REGIONS for analysis in _ANALYSES]
    group_labels = [f"{_REGION_LABELS[r]}\n{_ANALYSIS_LABELS[a]}" for r, a in groups]

    depths = list(_CHECKPOINT_DIRS.keys())
    bar_width = 0.35
    x = range(len(groups))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, depth in enumerate(depths):
        offset = (i - (len(depths) - 1) / 2) * bar_width
        heights = []
        for region, analysis in groups:
            row = sub[(sub["depth"] == depth) & (sub["region"] == region) & (sub["analysis"] == analysis)]
            heights.append(row["score"].iloc[0] if not row.empty else 0)
        ax.bar(
            [xi + offset for xi in x], heights, width=bar_width,
            color=_DEPTH_COLORS[depth], label=depth,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("score (mean across subjects)")
    ax.set_title(f"Depth 1 vs Depth 8 -- epoch {epoch}")
    ax.legend(frameon=False)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_scores()

    for depth in _CHECKPOINT_DIRS:
        plot_progression(df, depth, f"{depth}_progression.png")

    for epoch in _EPOCHS:
        plot_epoch_comparison(df, epoch, f"epoch{epoch}_comparison.png")
