"""Plot manifold-geometry metrics (SNR, capacity; analysis/manifold.py) as a
function of depth, for the transitions1/untiednorm/expansion8/50ep sweep.

Produces:
  - manifold_vs_depth_embedding.png: SNR and capacity at the `embedding`
    layer (the one layer every depth has in common) vs. depth (1/2/4/8).
  - manifold_vs_layer_by_depth.png: SNR and capacity vs. normalized layer
    position (fraction of the way through the tied backbone, "embedding"
    shown one step past the last iteration), one line per depth.

Usage:
    python plot_manifold_metrics.py
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_RESULTS_DB_PATH = Path("results.db")

_CHECKPOINT_DIRS = {
    "depth1": "model_checkpoints/ectiednet_imagenet1k_depth1_transitions1_untiednorm_expansion8_50ep",
    "depth2": "model_checkpoints/ectiednet_imagenet1k_depth2_transitions1_untiednorm_expansion8_50ep",
    "depth4": "model_checkpoints/ectiednet_imagenet1k_depth4_transitions1_untiednorm_expansion8_50ep",
    "depth8": "model_checkpoints/ectiednet_imagenet1k_depth8_transitions1_untiednorm_expansion8_50ep",
}
_DEPTH_VALUES = {"depth1": 1, "depth2": 2, "depth4": 4, "depth8": 8}
_EPOCH = 50
_METRICS = ["snr", "capacity"]
_METRIC_LABELS = {"snr": "SNR", "capacity": "Capacity"}

# Sequential blue ramp (dataviz skill palette.md, ordinal steps) -- depth is
# an ORDERED quantity (1<2<4<8), not an unordered category, so light-to-dark
# one-hue encodes it correctly instead of 4 arbitrary categorical hues.
_DEPTH_COLORS = {
    "depth1": "#86b6ef",   # step 250
    "depth2": "#5598e7",   # step 350
    "depth4": "#2a78d6",   # step 450 (slot-1 base blue)
    "depth8": "#104281",   # step 650
}


def load_manifold_scores() -> pd.DataFrame:
    conn = sqlite3.connect(str(_RESULTS_DB_PATH))
    placeholders = ",".join("?" * len(_CHECKPOINT_DIRS))
    df = pd.read_sql_query(
        f"SELECT checkpoint_dir, epoch, layer, compare_method AS metric, score "
        f"FROM results WHERE analysis = 'manifold' AND checkpoint_dir IN ({placeholders})",
        conn, params=list(_CHECKPOINT_DIRS.values()),
    )
    conn.close()
    if df.empty:
        raise ValueError(f"No analysis='manifold' rows in {_RESULTS_DB_PATH} for the depth 1/2/4/8 checkpoint dirs")

    dir_to_depth = {v: k for k, v in _CHECKPOINT_DIRS.items()}
    df["depth"] = df["checkpoint_dir"].map(dir_to_depth)
    return df


def _style_axis(ax):
    ax.grid(alpha=0.3, color="#e1e0d9")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")


def plot_vs_depth_embedding(df: pd.DataFrame, epoch: int, out_path: str):
    """SNR and capacity at the `embedding` layer vs. depth -- the one layer
    every depth has in common, so it's the most directly comparable point."""
    sub = df[(df["epoch"] == epoch) & (df["layer"] == "embedding")]
    depths_sorted = sorted(_CHECKPOINT_DIRS, key=lambda d: _DEPTH_VALUES[d])
    x_values = [_DEPTH_VALUES[d] for d in depths_sorted]

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(6 * len(_METRICS), 4.5))
    for ax, metric in zip(axes, _METRICS):
        line = sub[sub["metric"] == metric].set_index("depth").reindex(depths_sorted)
        missing = line["score"].isna()
        if missing.any():
            print(f"  [warn] embedding/{metric}: no epoch-{epoch} data for {[d for d, m in zip(depths_sorted, missing) if m]}")
        ax.plot(x_values, line["score"], marker="o", markersize=6, linewidth=2, color="#2a78d6")
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_xlabel("depth")
        ax.set_ylabel(f"{_METRIC_LABELS[metric]} (embedding layer)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(v) for v in x_values])
        _style_axis(ax)

    fig.suptitle(f"Manifold geometry at the embedding layer vs. depth -- epoch {epoch}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _layer_position(layer: str, num_iterations: int) -> float:
    """Fraction of the way through the tied backbone; `embedding` (computed
    one step after the last iteration) is placed just past 1.0."""
    if layer == "embedding":
        return 1.15
    return int(layer.replace("iter", "")) / num_iterations


def plot_vs_layer_by_depth(df: pd.DataFrame, epoch: int, out_path: str):
    """SNR and capacity vs. normalized layer position, one line per depth."""
    sub = df[df["epoch"] == epoch]
    depths_sorted = sorted(_CHECKPOINT_DIRS, key=lambda d: _DEPTH_VALUES[d])

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(6 * len(_METRICS), 4.5))
    for ax, metric in zip(axes, _METRICS):
        for depth in depths_sorted:
            line = sub[(sub["depth"] == depth) & (sub["metric"] == metric)].copy()
            if line.empty:
                continue
            line["position"] = line["layer"].apply(lambda l: _layer_position(l, _DEPTH_VALUES[depth]))
            line = line.sort_values("position")
            ax.plot(
                line["position"], line["score"],
                marker="o", markersize=5, linewidth=2,
                color=_DEPTH_COLORS[depth], label=depth,
            )
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_xlabel("layer position (fraction of depth; rightmost = embedding)")
        ax.set_ylabel(_METRIC_LABELS[metric])
        ax.legend(frameon=False, title="depth")
        _style_axis(ax)

    fig.suptitle(f"Manifold geometry vs. layer position, by depth -- epoch {epoch}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_manifold_scores()
    plot_vs_depth_embedding(df, epoch=_EPOCH, out_path="manifold_vs_depth_embedding.png")
    plot_vs_layer_by_depth(df, epoch=_EPOCH, out_path="manifold_vs_layer_by_depth.png")
