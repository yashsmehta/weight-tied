"""RSA / encoding-score evaluation against NSD, orchestration layer.

Ported from visreps/evals.py, trimmed to the unified NSD multi-subject path.
Drops the things-behavior branch, the nsd_synthetic branch (`_eval_rsa_nsd_synthetic`,
`_lookup_nsd_best_layers`), and `reconstruct_from_pcs` (all PCA-label-only,
unused by this project).
"""
import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig, OmegaConf

import model_utils as mutils
from analysis.alignment import AlignmentData, _align_stimulus_level, compute_traintest_alignment, prepare_traintest_alignment
from analysis.rsa import compute_rdm, compute_rdm_correlation
from dataloaders.imagenet import get_transform
from dataloaders.nsd import _make_loader, load_all_nsd_data
from results import save_results
from utils import get_seed_letter, rprint


def _load_cfg(cfg):
    """Merge runtime cfg with training cfg (drops `mode`)."""
    seed_letter = get_seed_letter(cfg.seed)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    base = OmegaConf.load(path)
    epoch = int(cfg.checkpoint_model.split("_")[-1].split(".")[0])
    base.epoch = epoch
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    return OmegaConf.merge(base, cfg)


def _build_header(cfg):
    """Compact one-line summary header for eval output."""
    analysis = cfg.get("analysis", "rsa").upper()
    seed = cfg.get("seed", "?")
    seed_letter = get_seed_letter(seed) if isinstance(seed, int) else "?"
    cfg_id = cfg.get("cfg_id", "?")
    epoch = cfg.get("epoch", "?")
    neural_dataset = cfg.get("neural_dataset", "?").upper()
    region = cfg.get("region", "")
    subject_idx = cfg.get("subject_idx", "")

    parts = [f"{analysis} eval", f"cfg{cfg_id}{seed_letter} epoch {epoch}"]
    if region and str(region).upper() != "N/A":
        parts.append(f"{neural_dataset} {region}")
    else:
        parts.append(neural_dataset)
    if subject_idx != "" and str(subject_idx).upper() != "N/A":
        parts.append(f"subj {subject_idx}")
    parts.append(f"seed {seed}")
    return " | ".join(parts)


def _listify(val):
    """Ensure val is a plain Python list (handles int, str, ListConfig, list)."""
    if isinstance(val, (list, ListConfig)):
        return list(val)
    return [val]


def eval(cfg):
    """Unified NSD evaluation: one forward pass, per-subject per-region results.

    Accepts list-valued cfg.subject_idx and cfg.region: loads all NSD data
    once, extracts activations once, then iterates over all (subject, region)
    pairs internally.
    """
    verbose = cfg.get("verbose", False)

    if cfg.load_model_from == "checkpoint":
        cfg = _load_cfg(cfg)
    elif cfg.load_model_from == "torchvision":
        cfg.epoch = -1
        cfg.cfg_id = "pretrained" if cfg.pretrained_dataset == "imagenet1k" else "untrained"
        cfg.return_nodes = mutils.TORCHVISION_RETURN_NODES[cfg.model_name]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = _listify(cfg.subject_idx)
    regions = _listify(cfg.region)

    seed_letter = get_seed_letter(cfg.seed) if isinstance(cfg.seed, int) else "?"
    rprint(
        f"\n  {cfg.get('analysis', 'rsa').upper()} eval | cfg{cfg.get('cfg_id', '?')}{seed_letter} "
        f"epoch {cfg.get('epoch', '?')} | {cfg.neural_dataset.upper()} | "
        f"{len(subjects)} subjects x {len(regions)} regions | seed {cfg.seed}\n",
        style="info",
    )

    model = mutils.load_model(cfg, dev, verbose=verbose)
    model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)

    all_data = load_all_nsd_data(cfg, subjects=subjects, regions=regions)

    stimuli = all_data["stimuli"]
    rprint(
        f"  {len(subjects)} subjects x {len(regions)} regions, "
        f"{len(stimuli)} stimuli, {len(all_data['shared_test_ids'])} shared test IDs",
        style="success",
    )

    transform = get_transform(ds_stats="imgnet")
    dl = _make_loader(stimuli, transform, cfg.batchsize, cfg.num_workers)
    acts, ids = mutils.get_activations(model, dl, dev)
    rprint("  Activations extracted once for all subjects/regions", style="success")
    del dl

    analysis = cfg.get("analysis", "rsa").lower()
    if analysis == "rsa":
        results = _eval_rsa(cfg, model, acts, ids, all_data, subjects, regions, dev, verbose)
    elif analysis == "encoding_score":
        results = _eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose)
    else:
        raise ValueError(f"Unknown analysis method: {analysis}")

    torch.cuda.empty_cache()
    return results


def _eval_rsa(cfg, model, acts, ids, all_data, subjects, regions, dev, verbose):
    """Two-phase RSA: layer selection with SRP, then re-extract without SRP.

    Phase 1: per-(region, subject) layer selection using SRP activations.
    Phase 2: re-extract unique best layers without SRP for exact test RDMs,
             then compute per-subject point estimates + optional bootstrap CIs.
    """
    method = cfg.get("compare_method", "spearman").lower()
    bootstrap = cfg.get("bootstrap", False)
    n_bootstrap = cfg.get("n_bootstrap", 1000)

    # Subsample train stimuli for layer selection to keep RDM computation fast
    # (NSD has ~9k train stimuli per subject).
    n_select = cfg.get("n_select", 1000)

    neural = all_data["neural"]
    shared_test_ids = all_data["shared_test_ids"]
    stimuli = all_data["stimuli"]

    # ── Phase 1: layer selection (SRP acts) ──
    rprint("\n  Phase 1: Per-subject layer selection", style="info")
    per_region_layers = {}
    per_region_scores = {}

    for region in regions:
        per_region_layers[region] = {}
        per_region_scores[region] = {}
        for subj in subjects:
            subj_neural_train = neural[region][subj]["train"]
            train_acts, train_neural, _ = _align_stimulus_level(acts, subj_neural_train, ids)

            n_train_subj = train_neural.size(0)
            if n_select is not None and n_select < n_train_subj:
                rng_sel = np.random.RandomState(42)
                sel_idx = rng_sel.choice(n_train_subj, size=n_select, replace=False)
            else:
                sel_idx = np.arange(n_train_subj)
            neural_rdm_sel = compute_rdm(train_neural[sel_idx])

            best_layer, best_score = None, -float("inf")
            subj_scores = []
            for layer, layer_acts in train_acts.items():
                flat = layer_acts[sel_idx].flatten(start_dim=1) if layer_acts.ndim > 2 else layer_acts[sel_idx]
                layer_rdm = compute_rdm(flat)
                score = compute_rdm_correlation(layer_rdm, neural_rdm_sel, correlation=method.capitalize())
                subj_scores.append({"layer": layer, "score": score})
                if score > best_score:
                    best_score = score
                    best_layer = layer

            per_region_layers[region][subj] = best_layer
            per_region_scores[region][subj] = subj_scores

            if verbose:
                rprint(f"    {region} subj {subj}: {best_layer} ({best_score:.4f}), {len(sel_idx)} stimuli for selection", style="info")

            del train_acts, train_neural

    del acts
    torch.cuda.empty_cache()
    rprint("  Freed bulk SRP activations", style="success")

    # ── Phase 2: re-extract unique layers for test stimuli ──
    rprint("\n  Phase 2: Test evaluation", style="info")

    test_stimuli = {sid: stimuli[sid] for sid in shared_test_ids if sid in stimuli}
    transform = get_transform(ds_stats="imgnet")
    dl_test = _make_loader(test_stimuli, transform, cfg.batchsize, cfg.num_workers)
    rprint(f"  Test dataloader: {len(test_stimuli)} stimuli", style="success")

    all_unique_layers = set()
    for region_layers in per_region_layers.values():
        all_unique_layers.update(region_layers.values())

    model_rdms = {}
    for layer in sorted(all_unique_layers):
        rprint(f"  Re-extracting {layer} without SRP...", style="info")
        exact_acts, _ = mutils.extract_single_layer(model, dl_test, dev, layer, shared_test_ids)
        flat = exact_acts.flatten(start_dim=1) if exact_acts.ndim > 2 else exact_acts
        model_rdms[layer] = compute_rdm(flat)
        del exact_acts

    del model, dl_test
    torch.cuda.empty_cache()

    # ── Per-(region, subject) scoring + save ──
    all_results = []
    for region in regions:
        rprint(f"\n  -- Region: {region} --", style="info")

        for subj in subjects:
            best_layer = per_region_layers[region][subj]
            selection_scores = per_region_scores[region][subj]

            test_neural_dict = neural[region][subj]["test"]
            responses = [test_neural_dict[sid] for sid in shared_test_ids if sid in test_neural_dict]
            neural_tensor = torch.as_tensor(np.stack(responses).squeeze(), dtype=torch.float32)
            neural_rdm = compute_rdm(neural_tensor)

            point_estimate = compute_rdm_correlation(model_rdms[best_layer], neural_rdm, correlation=method.capitalize())

            ci_low, ci_high = None, None
            bootstrap_scores_list = None

            if bootstrap:
                rng = np.random.RandomState(42)
                n_test = neural_rdm.size(0)
                n_sub = int(n_test * 0.9)
                bootstrap_scores = np.empty(n_bootstrap, dtype=np.float64)

                for i in range(n_bootstrap):
                    idx = torch.from_numpy(rng.choice(n_test, size=n_sub, replace=False)).to(neural_rdm.device)
                    m = model_rdms[best_layer][idx][:, idx]
                    n = neural_rdm[idx][:, idx]
                    bootstrap_scores[i] = compute_rdm_correlation(m, n, correlation=method.capitalize())

                ci_low = float(np.percentile(bootstrap_scores, 2.5))
                ci_high = float(np.percentile(bootstrap_scores, 97.5))
                bootstrap_scores_list = bootstrap_scores.tolist()

            msg = f"    subj {subj} | {method.capitalize():<10}| {best_layer} = {point_estimate:.4f}"
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

            results_df = pd.DataFrame([result])

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(cfg, {"subject_idx": subj, "region": region})
                save_results(results_df, save_cfg)

            all_results.append(result)

    return pd.DataFrame(all_results)


def _eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose):
    """Per-(region, subject) encoding score using SRP activations throughout
    (unlike RSA, no re-extraction is needed)."""
    neural = all_data["neural"]

    all_results = []
    for region in regions:
        rprint(f"\n  -- Region: {region} --", style="info")

        for subj in subjects:
            subj_neural = neural[region][subj]

            train_data, test_data = prepare_traintest_alignment(cfg, acts, subj_neural, ids)

            alignment_scores = compute_traintest_alignment(cfg, train_data, test_data, verbose=verbose, re_extract_fn=None)

            del train_data, test_data
            torch.cuda.empty_cache()

            results_df = pd.DataFrame(alignment_scores)

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(cfg, {"subject_idx": subj, "region": region})
                save_results(results_df, save_cfg)

            all_results.extend(alignment_scores)

    del acts, model
    torch.cuda.empty_cache()

    return pd.DataFrame(all_results)
