"""Plot manifold-geometry metrics (SNR, capacity; analysis/manifold.py) as a
function of depth, for the transitions1/untiednorm/expansion8/50ep sweep.

Produces:
  - manifold_vs_depth_early_ventral.png: PRIMARY figure, structured exactly
    like depth_progression_epoch50.png (RSA/encoding) for visual consistency
    -- two panels (SNR, Capacity), each with an "Early" and "Ventral" line
    vs. depth (1/2/4/8). "Early"/"Ventral" here are proxy layers, not real
    brain regions (manifold analysis has no NSD data) -- Early reuses the
    exact layer that wins RSA's early-visual-stream layer selection at each
    depth (iter1/iter1/iter2/iter4 for depth 1/2/4/8, i.e. the last iteration
    before the single BlurPool transition), Ventral is the `embedding` layer
    (which wins RSA/encoding's ventral-stream selection at every depth,
    unanimously, across the whole sweep -- see PROGRESS.md session 9).
  - manifold_vs_layer_by_depth.png: SUPPLEMENTARY -- SNR and capacity vs.
    normalized layer position (fraction of the way through the tied
    backbone, "embedding" shown one step past the last iteration), one line
    per depth. Geometry stays roughly flat across iterations, then jumps
    sharply at the post-GAP embedding step, for every depth.

Also prints the exact pulled values (not eyeballed off a chart) for both
figures to stdout.

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

# The RSA-early-stream-winning layer at each depth (last iteration before the
# single num_transitions=1 BlurPool downsample) -- see PROGRESS.md session 9
# for the transition-point derivation. Reused here as the "Early" proxy so
# this figure lines up with the RSA/encoding early-stream layer choice.
_EARLY_LAYER_BY_DEPTH = {"depth1": "iter1", "depth2": "iter1", "depth4": "iter2", "depth8": "iter4"}
_VENTRAL_LAYER = "embedding"

# Same two roles, same colors as plot_depth_comparison.py's _REGION_COLORS --
# visual consistency with the RSA figure this replaces the table next to.
_ROLE_COLORS = {"Early": "#2a78d6", "Ventral": "#eb6834"}

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


def pull_early_ventral_values(df: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """Exact SNR/capacity values at the Early-proxy and Ventral (embedding)
    layers, one row per (depth, metric, role) -- the numbers the primary
    figure plots, pulled straight from results.db rather than read off a
    chart."""
    sub = df[df["epoch"] == epoch]
    depths_sorted = sorted(_CHECKPOINT_DIRS, key=lambda d: _DEPTH_VALUES[d])

    rows = []
    for depth in depths_sorted:
        roles = {"Early": _EARLY_LAYER_BY_DEPTH[depth], "Ventral": _VENTRAL_LAYER}
        for role, layer in roles.items():
            for metric in _METRICS:
                match = sub[(sub["depth"] == depth) & (sub["layer"] == layer) & (sub["metric"] == metric)]
                score = match["score"].iloc[0] if not match.empty else None
                rows.append({
                    "depth": _DEPTH_VALUES[depth], "role": role, "layer": layer,
                    "metric": _METRIC_LABELS[metric], "score": score,
                })
    values = pd.DataFrame(rows)
    missing = values[values["score"].isna()]
    if not missing.empty:
        print(f"  [warn] missing epoch-{epoch} values:\n{missing[['depth', 'role', 'layer', 'metric']].to_string(index=False)}")
    return values


def plot_early_ventral_vs_depth(values: pd.DataFrame, epoch: int, out_path: str):
    """PRIMARY figure: SNR and capacity vs. depth, Early vs. Ventral-proxy
    lines -- same structure as depth_progression_epoch50.png."""
    depths_sorted = sorted(_DEPTH_VALUES.values())

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(6 * len(_METRICS), 4.5))
    for ax, metric in zip(axes, _METRICS):
        metric_label = _METRIC_LABELS[metric]
        for role in ("Early", "Ventral"):
            line = values[(values["metric"] == metric_label) & (values["role"] == role)]
            line = line.set_index("depth").reindex(depths_sorted)
            ax.plot(
                depths_sorted, line["score"],
                marker="o", markersize=6, linewidth=2,
                color=_ROLE_COLORS[role], label=role,
            )
        ax.set_title(metric_label)
        ax.set_xlabel("depth")
        ax.set_ylabel(metric_label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths_sorted)
        ax.set_xticklabels([str(v) for v in depths_sorted])
        ax.legend(frameon=False)
        _style_axis(ax)

    fig.suptitle(f"Manifold geometry vs. depth (Early/Ventral proxy layers) -- epoch {epoch}")
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

    values = pull_early_ventral_values(df, epoch=_EPOCH)
    print("\nExact values plotted in manifold_vs_depth_early_ventral.png:")
    print(values.pivot_table(index="depth", columns=["metric", "role"], values="score").to_string())
    plot_early_ventral_vs_depth(values, epoch=_EPOCH, out_path="manifold_vs_depth_early_ventral.png")

    plot_vs_layer_by_depth(df, epoch=_EPOCH, out_path="manifold_vs_layer_by_depth.png")
