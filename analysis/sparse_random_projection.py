"""Sparse random projection for compressing activations before RSA/encoding-score.

Ported verbatim from visreps/analysis/sparse_random_projection.py -- despite
appearing PCA/experiment-adjacent by name, this is unconditionally called by
model_utils.get_activations for every RSA/encoding-score run, not just PCA-label
experiments.
"""
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.random_projection import SparseRandomProjection

from utils import rprint


def _validate_transformer(
    transformer: SparseRandomProjection,
    k: int,
    requested_density: Optional[float],
    seed: Optional[int],
) -> bool:
    """Check if a loaded SparseRandomProjection transformer matches the required params."""
    if transformer.n_components != k:
        rprint(f"Cached transformer k mismatch (Loaded: {transformer.n_components}, Requested: {k}).", style="warning")
        return False

    loaded_density_value = getattr(transformer, "density_", None)
    if loaded_density_value is None:
        rprint("Cached transformer seems invalid (no density_ attribute after fit).", style="warning")
        return False

    if requested_density is not None and not np.isclose(loaded_density_value, requested_density):
        rprint(f"Cached transformer density mismatch (Loaded: {loaded_density_value:.4f}, Requested: {requested_density:.4f}).", style="warning")
        return False

    loaded_seed = getattr(transformer, "random_state", None)
    if loaded_seed != seed:
        rprint(f"Cached transformer seed mismatch (Loaded: {loaded_seed}, Requested: {seed}).", style="warning")
        return False

    return True


def _fit_new_transformer(
    D: int,
    k: int,
    density: Optional[float],
    seed: Optional[int],
) -> Optional[SparseRandomProjection]:
    """Fit a new SparseRandomProjection transformer with the specified parameters."""
    density_param = density if density is not None else "auto"
    rprint(f"Fitting SRP (D={D}->k={k})", style="info")

    try:
        transformer = SparseRandomProjection(n_components=k, density=density_param, random_state=seed)
        dummy_data = np.zeros((1, D), dtype=np.float32)
        transformer.fit(dummy_data)
        return transformer
    except Exception as e:
        rprint(f"Failed to fit SRP transformer: {e}", style="error")
        return None


def get_srp_transformer(
    D: int,
    k: int,
    density: Optional[float],
    seed: Optional[int],
    cache_dir: str,
) -> Optional[SparseRandomProjection]:
    """Retrieve a cached SparseRandomProjection transformer or fit + cache a new one."""
    if k <= 0 or D <= 0:
        rprint(f"Invalid dimensions D={D}, k={k}. Cannot create transformer.", style="error")
        return None

    os.makedirs(cache_dir, exist_ok=True)

    density_str = f"{density:.4f}" if density is not None else "auto"
    transformer_filename = f"srp_D{D}_k{k}_density{density_str}_seed{seed}.joblib"
    transformer_path = os.path.join(cache_dir, transformer_filename)

    transformer: Optional[SparseRandomProjection] = None
    should_fit_new = False

    if os.path.exists(transformer_path):
        try:
            loaded_transformer = joblib.load(transformer_path)
            if _validate_transformer(loaded_transformer, k, density, seed):
                transformer = loaded_transformer
            else:
                rprint("Cached transformer validation failed. Will refit.", style="warning")
                should_fit_new = True
        except Exception as e:
            rprint(f"Error loading cached transformer: {e}. Will refit.", style="warning")
            should_fit_new = True
            try:
                os.remove(transformer_path)
            except OSError as remove_err:
                rprint(f"Could not remove problematic cache file: {remove_err}", style="warning")
    else:
        should_fit_new = True

    if should_fit_new:
        transformer = _fit_new_transformer(D, k, density, seed)
        if transformer is not None:
            try:
                joblib.dump(transformer, transformer_path)
            except Exception as e:
                rprint(f"Failed to cache transformer: {e}", style="warning")

    return transformer
