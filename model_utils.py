"""Model loading, checkpointing, and activation extraction.

Ported from visreps/models/utils.py, trimmed to the two models this project
uses (ECTiedNet, AlexNet) -- drops CustomCNN/TinyCustomCNN and the
custom_model config path entirely. Checkpoint provenance (git SHA capture)
folded in from visreps/git_sha_capture.patch.
"""
import json
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from analysis.sparse_random_projection import get_srp_transformer
from model import ECTiedNet
from utils import console, get_seed_letter, rprint

# Default extraction points for pretrained torchvision models.
TORCHVISION_RETURN_NODES = {
    "AlexNet": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
}


class FeatureExtractor(nn.Module):
    """Generic hook-based activation extractor (for AlexNet; ECTiedNet self-reports
    via its `return_nodes` dict-mode forward instead -- a plain hook on a weight-tied
    module only ever captures its LAST call).
    """

    def __init__(self, model: nn.Module, return_nodes: Dict[str, str] = None,
                 post_relu: bool = True, extract_pre_and_post: bool = True):
        super().__init__()
        self.model = model
        if isinstance(return_nodes, list):
            return_nodes = {node: node for node in return_nodes}
        self.return_nodes = return_nodes
        self.post_relu = post_relu
        self.extract_pre_and_post = extract_pre_and_post
        self.features = {}
        self.handles = []

        base_mapping = self._create_layer_mapping()

        if self.extract_pre_and_post:
            post_mapping = self._remap_to_post_relu(base_mapping)
            self.layer_mapping, self.return_nodes = self._build_pre_post_mapping(base_mapping, post_mapping)
        elif self.post_relu:
            self.layer_mapping = self._remap_to_post_relu(base_mapping)
        else:
            self.layer_mapping = base_mapping

        self._attach_hooks()

    def _create_layer_mapping(self):
        """Map semantic names (conv1, fc1) to actual layer paths."""
        mapping = {}
        conv_count = 1
        fc_count = 1

        def is_conv(module):
            return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))

        def is_fc(module):
            return isinstance(module, nn.Linear)

        def is_main_conv(name, module):
            return is_conv(module) and "downsample" not in name

        seen_modules = set()

        if isinstance(self.model, torchvision.models.resnet.ResNet):
            if hasattr(self.model, "conv1"):
                mapping["conv1"] = "conv1"
                conv_count += 1
                seen_modules.add(id(self.model.conv1))
            block_count = 1
            for block_name in ["layer1", "layer2", "layer3", "layer4"]:
                if hasattr(self.model, block_name):
                    block = getattr(self.model, block_name)
                    for sub_block_idx, sub_block in enumerate(block):
                        mapping[f"block{block_count}"] = f"{block_name}.{sub_block_idx}"
                        block_count += 1
                        seen_modules.add(id(sub_block))
            if hasattr(self.model, "fc"):
                mapping["fc1"] = "fc"
                seen_modules.add(id(self.model.fc))

        elif isinstance(self.model, torchvision.models.VisionTransformer):
            if hasattr(self.model, "conv_proj"):
                mapping["patch_embed"] = "conv_proj"
                seen_modules.add(id(self.model.conv_proj))
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                for i, block in enumerate(self.model.encoder.layers):
                    mapping[f"block{i + 1}"] = f"encoder.layers.encoder_layer_{i}"
                    seen_modules.add(id(block))
            if hasattr(self.model, "heads") and hasattr(self.model.heads, "head"):
                mapping["head"] = "heads.head"
                seen_modules.add(id(self.model.heads.head))

        elif hasattr(self.model, "features") and hasattr(self.model, "classifier"):
            # AlexNet/VGG style
            for name, module in self.model.features.named_modules():
                if is_conv(module) and id(module) not in seen_modules:
                    mapping[f"conv{conv_count}"] = f"features.{name}"
                    conv_count += 1
                    seen_modules.add(id(module))
            for name, module in self.model.classifier.named_modules():
                if is_fc(module) and id(module) not in seen_modules:
                    mapping[f"fc{fc_count}"] = f"classifier.{name}"
                    fc_count += 1
                    seen_modules.add(id(module))

        if not mapping:
            for name, module in self.model.named_modules():
                if id(module) in seen_modules:
                    continue
                if is_main_conv(name, module):
                    mapping[f"conv{conv_count}"] = name
                    conv_count += 1
                    seen_modules.add(id(module))
                elif is_fc(module):
                    mapping[f"fc{fc_count}"] = name
                    fc_count += 1
                    seen_modules.add(id(module))

        return mapping

    def _remap_to_post_relu(self, mapping):
        """Remap Conv2d/Linear paths to their downstream activation fn (post-ReLU)."""
        relu_mapping = {}
        requested = set(self.return_nodes or [])
        for semantic_name, module_path in mapping.items():
            if requested and semantic_name not in requested:
                relu_mapping[semantic_name] = module_path
                continue
            parts = module_path.split(".")
            if len(parts) == 2:
                container_name, idx_str = parts
                container = getattr(self.model, container_name, None)
                if container is not None and isinstance(container, nn.Sequential):
                    try:
                        module_idx = int(idx_str)
                    except ValueError:
                        relu_mapping[semantic_name] = module_path
                        continue
                    found = False
                    for i in range(module_idx + 1, len(container)):
                        if isinstance(container[i], (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                            relu_mapping[semantic_name] = f"{container_name}.{i}"
                            found = True
                            break
                        if isinstance(container[i], (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                            break
                    if not found:
                        relu_mapping[semantic_name] = module_path
                else:
                    relu_mapping[semantic_name] = module_path
            else:
                relu_mapping[semantic_name] = module_path
        return relu_mapping

    def _build_pre_post_mapping(self, base_mapping, post_mapping):
        """Expand into _pre (raw conv/linear output) and _post (post-BN/ReLU) entries."""
        combined_mapping = {}
        expanded_return_nodes = {}

        for semantic_name, output_name in (self.return_nodes or {}).items():
            base_path = base_mapping.get(semantic_name)
            post_path = post_mapping.get(semantic_name)

            if base_path is None:
                rprint(f"Warning: {semantic_name} not found in base mapping", style="warning")
                continue

            if post_path is not None and base_path != post_path:
                pre_name, post_name = f"{semantic_name}_pre", f"{semantic_name}_post"
                combined_mapping[pre_name] = base_path
                combined_mapping[post_name] = post_path
                expanded_return_nodes[pre_name] = pre_name
                expanded_return_nodes[post_name] = post_name
            else:
                combined_mapping[semantic_name] = base_path
                expanded_return_nodes[semantic_name] = output_name

        return combined_mapping, expanded_return_nodes

    def _attach_hooks(self):
        actual_nodes = {}
        for semantic_name in self.return_nodes:
            if semantic_name in self.layer_mapping:
                actual_nodes[self.layer_mapping[semantic_name]] = self.return_nodes[semantic_name]
            else:
                rprint(f"Warning: {semantic_name} not found in model", style="warning")

        for name, module in self.model.named_modules():
            if name in actual_nodes:
                def get_hook(name):
                    def hook(module, input, output):
                        self.features[actual_nodes[name]] = output
                    return hook
                self.handles.append(module.register_forward_hook(get_hook(name)))

    def forward(self, x):
        self.features.clear()
        self.model(x)
        return self.features

    def __del__(self):
        for handle in self.handles:
            handle.remove()


def configure_feature_extractor(cfg, model, verbose=False):
    return_nodes = OmegaConf.to_container(cfg.get("return_nodes", {}), resolve=True)
    if not return_nodes:
        raise ValueError("return_nodes must be specified in config")
    return_nodes = {node: node for node in return_nodes} if isinstance(return_nodes, list) else return_nodes
    extract_pre_and_post = cfg.get("extract_pre_and_post", True)
    model.eval()
    extractor = FeatureExtractor(model, return_nodes, extract_pre_and_post=extract_pre_and_post)
    n_points = len(extractor.return_nodes)
    n_layers = len(return_nodes)
    suffix = f" ({n_layers} layers x pre/post)" if extract_pre_and_post else ""
    rprint(f"  {n_points} extraction points{suffix}", style="success")
    if verbose:
        rprint(f"    Layers: {list(return_nodes.keys())}", style="info")
        rprint(f"    Points: {list(extractor.return_nodes.keys())}", style="info")
    return extractor


def get_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List]:
    """Collect layer activations for every sample in `dataloader`.

    Always applies Sparse Random Projection (k=4096 per layer, capped by native
    dimensionality) to keep memory bounded.
    """
    model.eval()
    activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
    ids: List = []
    srp = {}

    try:
        probe_imgs, _ = next(iter(dataloader))
    except StopIteration:
        return {}, []

    with torch.no_grad():
        probe_out = model(probe_imgs.to(device))

    k_fixed, density, seed, cache_dir = 4096, None, None, "model_checkpoints/srp_cache"
    num_layers = len(probe_out)
    for name, out in probe_out.items():
        D = out.view(out.size(0), -1).size(1)
        transformer = get_srp_transformer(D=D, k=min(k_fixed, D), density=density, seed=seed, cache_dir=cache_dir)
        sparse_matrix = transformer.components_
        coo = sparse_matrix.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).float()
        proj_matrix = torch.sparse_coo_tensor(indices, values, coo.shape).to(device)
        srp[name] = proj_matrix

    rprint(f"  SRP transformers for {num_layers} layers (k={k_fixed})", style="success")

    with torch.no_grad(), Progress(
        TextColumn("  Extracting activations"), BarColumn(), MofNCompleteColumn(),
        TextColumn("batches"), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("extract", total=len(dataloader))
        for imgs, keys in dataloader:
            ids.extend(keys)
            feats = model(imgs.to(device))
            for name, out in feats.items():
                proj_matrix = srp.get(name)
                if proj_matrix is not None:
                    flat = out.view(out.size(0), -1).float()
                    out = torch.sparse.mm(proj_matrix, flat.t()).t()
                activations[name].append(out.cpu().float())
            progress.advance(task)

    return {n: torch.cat(b, 0) for n, b in activations.items()}, ids


def extract_single_layer(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layer_name: str,
    stimulus_ids: List[str] = None,
) -> Tuple[torch.Tensor, List]:
    """Re-extract one layer's full-resolution activations without SRP (post layer-selection)."""
    model.eval()
    all_acts = []
    all_ids = []

    with torch.no_grad(), Progress(
        TextColumn(f"  Re-extracting {layer_name}"), BarColumn(), MofNCompleteColumn(),
        TextColumn("batches"), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("re-extract", total=len(dataloader))
        for imgs, keys in dataloader:
            all_ids.extend(keys)
            feats = model(imgs.to(device))
            out = feats[layer_name]
            all_acts.append(out.view(out.size(0), -1).cpu().float())
            progress.advance(task)

    acts = torch.cat(all_acts, 0)

    if stimulus_ids is not None:
        id_to_idx = {str(k): i for i, k in enumerate(all_ids)}
        keep_idx = [id_to_idx[str(sid)] for sid in stimulus_ids if str(sid) in id_to_idx]
        acts = acts[keep_idx]
        all_ids = [all_ids[i] for i in keep_idx]

    rprint(f"  Re-extracted {layer_name}: {acts.shape} (exact, no SRP)", style="success")
    return acts, all_ids


def load_model(cfg, device, num_classes=None, verbose=False):
    """Load a model from checkpoint, or build a fresh ECTiedNet/AlexNet.

    cfg keys used:
      - eval, from checkpoint: load_model_from="checkpoint", seed, cfg_id,
        checkpoint_dir, checkpoint_model
      - train, or eval from torchvision: model_name ("ECTiedNet"/"AlexNet"),
        pretrained_dataset, arch (ECTiedNet only)
    """
    if getattr(cfg, "load_model_from", None) == "checkpoint":
        if num_classes is not None:
            rprint("WARNING: num_classes is ignored when loading from checkpoint", style="warning")
        seed_letter = get_seed_letter(cfg.seed)
        checkpoint_path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/{cfg.checkpoint_model}"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        rprint(f"  Loaded checkpoint (cfg{cfg.cfg_id}{seed_letter})", style="success")
        if verbose:
            rprint(f"    Path: {checkpoint_path}", style="info")
        return checkpoint["model"].to(device)

    model_name = getattr(cfg, "model_name", "AlexNet")
    pretrained_dataset = getattr(cfg, "pretrained_dataset", "none")

    if model_name == "ECTiedNet":
        # Depth-parameterized: takes an `arch` kwargs block instead of the
        # fixed (pretrained_dataset, num_classes) signature AlexNet uses.
        if pretrained_dataset != "none":
            raise ValueError(f"ECTiedNet has no pretrained weights; pretrained_dataset must be 'none', got '{pretrained_dataset}'")
        arch_cfg = getattr(cfg, "arch", {})
        arch_kwargs = OmegaConf.to_container(arch_cfg, resolve=True) if arch_cfg else {}
        model = ECTiedNet(num_classes=num_classes, stem="imagenet", **arch_kwargs)
    elif model_name == "AlexNet":
        if pretrained_dataset == "imagenet1k":
            model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        elif pretrained_dataset == "none":
            model = torchvision.models.alexnet(weights=None)
        else:
            raise ValueError(f"Invalid pretrained_dataset: {pretrained_dataset}")
        if num_classes is not None and num_classes != 1000:
            model.classifier[-1] = torch.nn.Linear(4096, num_classes)
            torch.nn.init.xavier_uniform_(model.classifier[-1].weight)
            torch.nn.init.zeros_(model.classifier[-1].bias)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


def _get_git_state():
    """Best-effort git SHA + dirty-tree flag for checkpoint provenance.

    Never raises -- training should not fail just because git info is
    unavailable (e.g. no git installed, or invoked outside a git checkout).
    """
    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


def setup_checkpoint_dir(cfg, model):
    """Create model_checkpoints/{cfg.checkpoint_dir}/cfg{num_classes}{seed_letter}/, save config.json."""
    seed_letter = get_seed_letter(cfg.seed)
    cfg_num = 1000
    subdir_name = f"cfg{cfg_num}{seed_letter}"
    checkpoint_path = os.path.join("model_checkpoints", cfg.checkpoint_dir, subdir_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    cfg_dict = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **_get_git_state(),
        **OmegaConf.to_container(cfg, resolve=True),
    }

    with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    return checkpoint_path, cfg_dict


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, metrics, cfg_dict):
    """Save model (whole nn.Module) + metrics + config at a checkpoint. Optimizer
    state is deliberately not saved (matches visreps)."""
    torch.save(
        {"epoch": epoch, "model": model, "metrics": metrics, "config": cfg_dict},
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"),
    )
