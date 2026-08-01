"""RSA (representational similarity analysis) core + train/test layer selection.

Ported from visreps/analysis/rsa.py, as-is aside from the import path and
dropping `_concept_average_exact` (THINGS-behavior-only, unreachable from the
NSD path this project uses).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import scipy.stats
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from utils import console, rprint

if TYPE_CHECKING:
    from analysis.alignment import AlignmentData

logger = logging.getLogger(__name__)


def _kendall_tau_a(x: np.ndarray, y: np.ndarray) -> tuple:
    """Kendall tau-a: (C - D) / n_pairs. No tie adjustment."""
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))

    tau_b = scipy.stats.kendalltau(x, y).statistic
    if np.isnan(tau_b):
        return (float("nan"), float("nan"))

    n0 = n * (n - 1) // 2
    t_x = sum(c * (c - 1) // 2 for c in np.unique(x, return_counts=True)[1])
    t_y = sum(c * (c - 1) // 2 for c in np.unique(y, return_counts=True)[1])

    denom = np.sqrt(np.float64(n0 - t_x) * np.float64(n0 - t_y))
    tau_a = float("nan") if denom == 0 else float(tau_b * denom / n0)
    return (tau_a, float("nan"))


_CORR_FUNCS = {
    "pearson": scipy.stats.pearsonr,
    "spearman": scipy.stats.spearmanr,
    "kendall": _kendall_tau_a,
}


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Row-wise dense ranking via double argsort (ties get consecutive ranks)."""
    return torch.argsort(torch.argsort(x, dim=1), dim=1).float()


def compute_rdm(
    representations: torch.Tensor, *, correlation: str = "Pearson", correction: float = 1e-12
) -> torch.Tensor:
    """(n_samples x n_samples) RDM (1 - correlation), Pearson or Spearman.

    Diagonal is 0 (zero self-dissimilarity); off-diagonal ranges 0 (identical)
    to 2 (perfectly anti-correlated).
    """
    corr = correlation.lower()
    if corr not in {"pearson", "spearman"}:
        raise ValueError("correlation must be 'Pearson' or 'Spearman'")

    x = representations.float().clone()
    if corr == "spearman":
        x = _rank(x)

    x -= x.mean(dim=1, keepdim=True)
    std = x.pow(2).mean(dim=1).add(correction).sqrt()

    zero_mask = std < correction * 10
    if zero_mask.any():
        logger.warning("%d/%d rows have ~zero variance for %s RDM", zero_mask.sum(), len(std), correlation)
        std[zero_mask] = 1.0

    cov = (x @ x.T) / x.size(1)
    corr_mat = cov / (std[:, None] * std[None, :] + correction)
    corr_mat.clamp_(-1, 1).fill_diagonal_(1.0)
    return 1.0 - corr_mat


def compute_rdm_correlation(
    rdm1: torch.Tensor, rdm2: torch.Tensor, *, correlation: str = "Kendall"
) -> float:
    """Correlation between two RDMs (Pearson/Spearman/Kendall). NaN if undefined."""
    if rdm1.shape != rdm2.shape or rdm1.ndim != 2:
        raise ValueError("RDMs must share the same 2-D shape")

    n = rdm1.size(0)
    if n <= 1:
        logger.warning("RDM dimension <= 1; correlation undefined")
        return float("nan")

    idx = torch.triu_indices(n, n, offset=1, device=rdm1.device)
    v1 = rdm1[idx[0], idx[1]].cpu().numpy()
    v2 = rdm2[idx[0], idx[1]].cpu().numpy()
    if v1.size == 0:
        return float("nan")

    corr = correlation.lower()
    if corr not in _CORR_FUNCS:
        raise ValueError("correlation must be 'Pearson', 'Spearman', or 'Kendall'")

    try:
        val, _ = _CORR_FUNCS[corr](v1, v2)
        if np.isnan(val):
            logger.warning("NaN returned for %s correlation", correlation)
            return float("nan")
        return float(val)
    except Exception as e:  # pragma: no cover
        logger.error("Error computing %s correlation: %s", correlation, e)
        return float("nan")


def compute_rsa(
    cfg: Dict,
    selection: "AlignmentData",
    evaluation: "AlignmentData",
    n_select: int | None = None,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    re_extract_fn=None,
) -> List[Dict]:
    """Train/test RSA: select best layer on train data, evaluate on the test set.

    1. Use all train stimuli (or subsample *n_select*) for layer selection.
    2. Build RDMs (Pearson) and pick the best-aligning layer.
    3. If *re_extract_fn* is given, re-extract the best layer without SRP for
       exact test RDMs; else use SRP'd activations.
    4. If *bootstrap*, subsample 90% of the eval set *n_bootstrap* times for CIs.
    """
    method = cfg.get("compare_method", "spearman").lower()
    rng = np.random.RandomState(seed)

    n_train = selection.neural.size(0)
    n_test = evaluation.neural.size(0)

    if n_select is not None and n_select < n_train:
        n_sel = n_select
        sel_idx = rng.choice(n_train, size=n_sel, replace=False)
        sel_label = f"subsampling {n_sel}"
    else:
        n_sel = n_train
        sel_idx = np.arange(n_train)
        sel_label = f"using all {n_sel}"

    if verbose:
        rprint(f"Train/test RSA: {n_train} train, {n_test} test, {sel_label} for layer selection", style="info")
        rprint(f"Building RDMs with Pearson, comparing with {method.capitalize()}", style="info")

    neural_rdm_sel = compute_rdm(selection.neural[sel_idx])

    selection_scores = []
    best_layer, best_score = None, -float("inf")
    for layer, acts in selection.activations.items():
        flat = acts[sel_idx].flatten(start_dim=1) if acts.ndim > 2 else acts[sel_idx]
        layer_rdm = compute_rdm(flat)
        score = compute_rdm_correlation(layer_rdm, neural_rdm_sel, correlation=method.capitalize())
        selection_scores.append({"layer": layer, "score": score})
        if verbose:
            rprint(f"  [select] {layer:<15} RSA = {score:.4f}", style="info")
        if score > best_score:
            best_score = score
            best_layer = layer

    if verbose:
        rprint(f"  Best layer: {best_layer} (score={best_score:.4f})", style="highlight")

    if re_extract_fn is not None:
        rprint(f"  Re-extracting {best_layer} without SRP for exact test RDMs...", style="info")
        exact_acts, _ = re_extract_fn(best_layer, evaluation.stimulus_ids)
        test_acts_flat = exact_acts.flatten(start_dim=1) if exact_acts.ndim > 2 else exact_acts
    else:
        test_acts = evaluation.activations[best_layer]
        test_acts_flat = test_acts.flatten(start_dim=1) if test_acts.ndim > 2 else test_acts

    test_neural_rdm = compute_rdm(evaluation.neural)
    test_model_rdm = compute_rdm(test_acts_flat)

    point_estimate = compute_rdm_correlation(test_model_rdm, test_neural_rdm, correlation=method.capitalize())
    if verbose:
        rprint(f"  Test RSA = {point_estimate:.4f}", style="highlight")

    ci_low, ci_high = None, None
    bootstrap_scores_list = None

    if bootstrap:
        n_subsample = int(n_test * 0.9)
        bootstrap_scores = np.empty(n_bootstrap, dtype=np.float64)

        progress = Progress(
            TextColumn("  Bootstrap             "), BarColumn(), MofNCompleteColumn(),
            TextColumn("iters"), TimeElapsedColumn(), console=console,
        )
        progress.start()
        boot_task = progress.add_task("bootstrap", total=n_bootstrap)
        for i in range(n_bootstrap):
            boot_idx = torch.from_numpy(rng.choice(n_test, size=n_subsample, replace=False)).to(test_neural_rdm.device)
            rdm_brain_boot = test_neural_rdm[boot_idx][:, boot_idx]
            rdm_model_boot = test_model_rdm[boot_idx][:, boot_idx]
            bootstrap_scores[i] = compute_rdm_correlation(rdm_model_boot, rdm_brain_boot, correlation=method.capitalize())
            progress.advance(boot_task)
        progress.stop()

        ci_low = float(np.percentile(bootstrap_scores, 2.5))
        ci_high = float(np.percentile(bootstrap_scores, 97.5))
        bootstrap_scores_list = bootstrap_scores.tolist()

    rprint("")
    msg = f"  {method.capitalize():<10}| {best_layer} = {point_estimate:.4f}"
    if bootstrap:
        msg += f"  [95% CI: {ci_low:.4f}, {ci_high:.4f}]"
    rprint(msg, style="highlight")

    result = {
        "layer": best_layer,
        "compare_method": method,
        "score": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "analysis": "rsa",
        "layer_selection_scores": selection_scores,
    }
    if bootstrap_scores_list is not None:
        result["bootstrap_scores"] = bootstrap_scores_list

    return [result]
