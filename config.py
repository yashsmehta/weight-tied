"""Lightweight config loader + validator for train/eval JSON configs.

Rewritten rather than ported from visreps/utils.py's ConfigVerifier, which
validates PCA-labels / custom_model / multi-neural-dataset options this
project never uses. Preserves the constraints visreps actually enforces
today (verified against its current source, not assumed from docs):
seed in {1,2,3}, NSD region strings, subject_idx in [0,7].
"""
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

from utils import get_seed_letter, rprint

VALID_DATASETS = {"imagenet", "imagenet-mini-10", "imagenet-mini-50", "imagenet-mini-200"}
VALID_MODEL_NAMES = {"AlexNet", "ECTiedNet"}
VALID_MODEL_SOURCES = {"torchvision", "checkpoint"}
VALID_ANALYSES = {"rsa", "encoding_score"}
VALID_COMPARE_METHODS = {"spearman", "kendall"}
VALID_NSD_REGIONS = {
    "early visual stream", "ventral visual stream",
    "V1", "V2", "V3", "hV4", "FFA", "PPA",
}


def merge_nested_config(cfg, source_key: str) -> None:
    """Flatten cfg[source_key] into the root config and drop the sibling key."""
    if source_key not in cfg:
        return
    source = OmegaConf.to_container(cfg[source_key], resolve=True)
    cfg.update(source)
    del cfg[source_key]


def load_config(config_path, overrides: list[str] | None = None):
    """Load a JSON config and apply `key=value` CLI overrides.

    For eval configs, merges the `torchvision` or `checkpoint` block into the
    root (whichever `load_model_from` selects) and drops the other.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    if cfg.mode == "eval":
        source_key = cfg.load_model_from
        other_key = {"torchvision": "checkpoint", "checkpoint": "torchvision"}[source_key]
        if other_key in cfg:
            del cfg[other_key]
        merge_nested_config(cfg, source_key)

    # Re-apply overrides so they win over whatever the nested block set.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    if cfg.mode == "eval" and cfg.load_model_from == "torchvision" and "cfg_id" in cfg:
        del cfg.cfg_id

    if cfg.get("verbose", False):
        rprint(f"Final Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}\n")
    return cfg


def _as_list(value):
    if isinstance(value, (list, ListConfig)):
        return list(value)
    return [value]


def validate_config(cfg):
    """Validate a loaded config, raising AssertionError on anything invalid."""
    if cfg.mode not in {"train", "eval"}:
        raise AssertionError(f"Invalid mode: {cfg.mode}. Must be 'train' or 'eval'")
    return _validate_train(cfg) if cfg.mode == "train" else _validate_eval(cfg)


def _validate_train(cfg):
    if cfg.dataset not in VALID_DATASETS:
        raise AssertionError(f"Invalid dataset: {cfg.dataset}. Must be one of {VALID_DATASETS}")
    if cfg.model_name not in VALID_MODEL_NAMES:
        raise AssertionError(f"Invalid model_name: {cfg.model_name}. Must be one of {VALID_MODEL_NAMES}")
    if not hasattr(cfg, "batchsize"):
        cfg.batchsize = 64
        rprint("Using default batch size: 64", style="info")
    if cfg.get("verbose", False):
        rprint("Config validated", style="success")
    return cfg


def _validate_eval(cfg):
    if getattr(cfg, "seed", None) not in (1, 2, 3):
        raise AssertionError(f"Invalid seed: {cfg.get('seed')}. Must be one of [1, 2, 3]")

    if cfg.neural_dataset.lower() != "nsd":
        raise AssertionError(f"Invalid neural_dataset: {cfg.neural_dataset}. Only 'nsd' is supported")

    subj = _as_list(cfg.subject_idx)
    cfg.subject_idx = subj
    for s in subj:
        if not isinstance(s, int) or not 0 <= s < 8:
            raise AssertionError(f"Invalid subject_idx for NSD: {s}. Must be an integer in [0, 7]")

    region = _as_list(cfg.region)
    cfg.region = region
    for r in region:
        if r not in VALID_NSD_REGIONS:
            raise AssertionError(f"Invalid region for NSD: {r}. Must be one of {VALID_NSD_REGIONS}")

    if cfg.analysis.lower() not in VALID_ANALYSES:
        raise AssertionError(f"Invalid analysis: {cfg.analysis}. Must be one of {VALID_ANALYSES}")

    if cfg.analysis.lower() == "encoding_score":
        # Encoding metric is always Pearson r; compare_method is an RSA concept.
        cfg.compare_method = "pearson"
    else:
        compare_method = cfg.get("compare_method", "spearman").lower()
        if compare_method not in VALID_COMPARE_METHODS:
            raise AssertionError(f"Invalid compare_method: {compare_method}. Must be one of {VALID_COMPARE_METHODS}")
        cfg.compare_method = compare_method

    if not hasattr(cfg.return_nodes, "__iter__") or not cfg.return_nodes:
        raise AssertionError("return_nodes must be a non-empty list-like object")

    if cfg.load_model_from not in VALID_MODEL_SOURCES:
        raise AssertionError(f"load_model_from must be one of {VALID_MODEL_SOURCES}")

    if cfg.load_model_from == "checkpoint":
        seed_letter = get_seed_letter(cfg.seed)
        checkpoint_path = Path(f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/{cfg.checkpoint_model}")
        if not checkpoint_path.exists():
            raise AssertionError(f"Checkpoint not found: {checkpoint_path}")

    if not cfg.get("log_expdata", False):
        rprint(
            "log_expdata is false — results will print to console but NOT be saved. "
            "Pass --override log_expdata=true to persist.",
            style="warning",
        )

    if cfg.get("verbose", False):
        rprint("Evaluation configuration validation successful", style="success")
    return cfg
