"""Train/test alignment data prep + dispatch to RSA or encoding-score.

Ported from visreps/analysis/alignment.py. Despite living next to RSA/PCA-
adjacent code, this is the actual train/test-split machinery evals.py uses
unconditionally for both RSA and encoding-score -- not PCA-specific.
Drops `prepare_concept_alignment` (THINGS-behavior-only, unreachable from
the NSD path this project uses).
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from analysis.encoding_score import compute_encoding_score
from analysis.rsa import compute_rsa

logger = logging.getLogger(__name__)


@dataclass
class AlignmentData:
    """Bundled activations and neural data for one split (train or test)."""

    activations: Dict[str, torch.Tensor]  # {layer_name: (n_stimuli, features)}
    neural: torch.Tensor  # (n_stimuli, n_voxels)
    stimulus_ids: Optional[List[str]] = None  # ordered IDs matching rows


def _align_stimulus_level(acts_raw, targets, keys):
    """Align activations with neural targets by stimulus ID.

    Returns (acts, neural, matched_ids) where matched_ids are the stimulus IDs
    in the same row order as acts/neural.
    """
    idx = [i for i, k in enumerate(keys) if str(k) in targets]
    matched_ids = [str(keys[i]) for i in idx]
    if not matched_ids:
        neural = torch.empty(0, dtype=torch.float32)
        acts = {l: a[:0] for l, a in acts_raw.items()}
        return acts, neural, matched_ids
    neural = torch.as_tensor(np.stack([targets[sid] for sid in matched_ids]), dtype=torch.float32)
    acts = {l: a[idx] for l, a in acts_raw.items()}
    return acts, neural, matched_ids


def prepare_traintest_alignment(
    cfg: DictConfig,
    acts_raw: Dict[str, torch.Tensor],
    neural_data_raw: Dict[str, Any],
    keys: List[str],
) -> Tuple[AlignmentData, AlignmentData]:
    """Align activations with train/test neural data (NSD stimulus-level split).

    Args:
        acts_raw: {layer: (n_total_stimuli, features)} from feature extraction.
        neural_data_raw: Must contain "train" and "test" keys mapping stimulus
            IDs to neural response vectors.
        keys: Stimulus IDs corresponding to rows of acts_raw.
    """
    train_acts, train_neural, train_ids = _align_stimulus_level(acts_raw, neural_data_raw["train"], keys)
    test_acts, test_neural, test_ids = _align_stimulus_level(acts_raw, neural_data_raw["test"], keys)
    train = AlignmentData(train_acts, train_neural, stimulus_ids=train_ids)
    test = AlignmentData(test_acts, test_neural, stimulus_ids=test_ids)

    logger.info(f"Prepared train/test alignment: {train.neural.size(0)} train, {test.neural.size(0)} test samples.")
    return train, test


def compute_traintest_alignment(
    cfg: DictConfig,
    train: AlignmentData,
    test: AlignmentData,
    verbose: bool = False,
    re_extract_fn=None,
) -> List[dict]:
    """Dispatch to RSA or encoding score based on cfg.analysis.

    re_extract_fn: optional callback (layer_name, stimulus_ids=None) -> (tensor, ids)
        for re-extracting a single layer without SRP. Passed to RSA only.
    """
    analysis = cfg.get("analysis", "rsa").lower()
    bootstrap = cfg.get("bootstrap", True)
    n_bootstrap = cfg.get("n_bootstrap", 1000)

    if analysis == "rsa":
        n_select = cfg.get("n_select", None)  # None = use all train stimuli
        return compute_rsa(
            cfg, train, test, n_select=n_select, bootstrap=bootstrap,
            n_bootstrap=n_bootstrap, verbose=verbose, re_extract_fn=re_extract_fn,
        )
    elif analysis == "encoding_score":
        return compute_encoding_score(train, test, bootstrap=bootstrap, n_bootstrap=n_bootstrap, verbose=verbose)
    else:
        raise ValueError(f"Unknown analysis method: {analysis}")
