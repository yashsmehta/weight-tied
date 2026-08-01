"""Leaf infrastructure: logging, env vars, accuracy/metrics, optimizer/scheduler setup.

Ported from visreps/utils.py, trimmed to what this project's train/eval pipeline
actually needs (no PCA-labels support, no custom_model config path).
"""
import csv
import os
import pickle
import sys
import warnings

import torch
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich.console import Console
from rich.theme import Theme
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    StepLR,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*",
)
warnings.filterwarnings("ignore", category=UserWarning, message="Corrupt EXIF data.*")


def is_interactive_environment():
    """True if running in a terminal/notebook rather than a batch job (e.g. SLURM)."""
    if os.environ.get("SLURM_JOB_ID") is not None:
        return False
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def setup_logging():
    """Initialize Rich with a custom theme and return (console, print) tuple."""
    custom_theme = Theme(
        {
            "info": "bold white",
            "success": "green",
            "warning": "bold yellow",
            "error": "bold red",
            "highlight": "bold magenta",
            "setup": "cyan",
        }
    )
    console = Console(theme=custom_theme)
    return console, console.print


console, rprint = setup_logging()


def get_env_var(key: str) -> str:
    """Get a required path from the environment (loads .env if present).

    Unlike visreps' version, this raises immediately on a missing var instead
    of silently returning "" — a missing env var should fail loudly here
    rather than surface later as a confusing FileNotFoundError.
    """
    load_dotenv()
    value = os.environ.get(key)
    if value is None:
        raise RuntimeError(f"Missing required env var: {key} (check your .env file)")
    return value


def load_pickle(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found at path: {file_path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError(f"Error unpickling file at {file_path}. File may be corrupted.")
    except Exception as e:
        raise RuntimeError(f"Error loading pickle file at {file_path}: {e}")


def get_seed_letter(seed: int) -> str:
    """Convert seed (1-9) to letter (a-i)."""
    if not isinstance(seed, int) or seed < 1 or seed > 9:
        raise ValueError(f"Seed must be an integer between 1-9, got {seed}")
    return chr(ord("a") + seed - 1)


def calculate_cls_accuracy(data_loader, model, device, criterion=None):
    """Compute top-1/top-5 accuracy (and optionally average loss) over a split.

    For models with fewer than 5 classes, top-5 accuracy is "" instead of a number.
    Returns (top1_acc, top5_acc, avg_loss) as percentages (0-100); avg_loss is
    None if no criterion was passed.
    """
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    total_loss = 0.0

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    use_top5 = None

    with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            batch_size = labels.size(0)
            total += batch_size

            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * batch_size

            if use_top5 is None:
                use_top5 = outputs.size(1) >= 5

            if not use_top5:
                _, preds = outputs.max(dim=1)
                top1_correct += (preds == labels).sum().item()
            else:
                maxk = 5
                _, preds = outputs.topk(maxk, dim=1, largest=True, sorted=True)
                preds = preds.t()
                correct = preds.eq(labels.to(preds.device).view(1, -1).expand_as(preds))
                top1_correct += correct[0].sum().item()
                top5_correct += correct.sum(dim=0).gt(0).sum().item()

    avg_loss = (total_loss / total) if (criterion is not None and total > 0) else None

    if total == 0:
        return 0.0, 0.0, avg_loss

    top1_acc = 100.0 * top1_correct / total
    if not use_top5:
        return top1_acc, "", avg_loss
    top5_acc = 100.0 * top5_correct / total
    return top1_acc, top5_acc, avg_loss


class MetricsLogger:
    """Logs training metrics to a per-run CSV and (optionally) wandb."""

    def __init__(self, cfg, checkpoint_dir=None):
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.metrics_file = None

        if checkpoint_dir:
            self.metrics_file = os.path.join(checkpoint_dir, "training_metrics.csv")
            fieldnames = [
                "epoch", "train_loss", "train_acc", "train_top5",
                "test_acc", "test_top5", "test_loss", "learning_rate",
            ]
            with open(self.metrics_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        self.use_wandb = cfg.get("use_wandb", False)
        if self.use_wandb:
            try:
                if not wandb.api.api_key:
                    rprint("WandB not authenticated. Run 'wandb login' first.", style="error")
                    self.use_wandb = False
                else:
                    os.environ["WANDB_SILENT"] = "true"
                    wandb.init(
                        project=cfg.dataset,
                        group=f"seed_{cfg.seed}",
                        name=cfg.get("checkpoint_dir", cfg.model_name),
                        config=OmegaConf.to_container(cfg, resolve=True),
                        tags=[cfg.model_name, f"lr_{cfg.learning_rate}"],
                        notes=f"Training {cfg.model_name} with seed {cfg.seed}",
                        settings=wandb.Settings(start_method="thread"),
                    )
                    wandb.define_metric("*", step_metric="epoch")
                    rprint(f"WandB initialized. View results at: {wandb.run.get_url()}", style="info")
            except Exception as e:
                rprint(f"W&B initialization failed: {e}", style="error")
                self.use_wandb = False

    def log_metrics(self, epoch, loss, metrics):
        if self.metrics_file:
            csv_metrics = {
                "epoch": metrics["epoch"],
                "train_loss": loss,
                "train_acc": metrics.get("train_acc", ""),
                "train_top5": metrics.get("train_top5", ""),
                "test_acc": metrics.get("test_acc", ""),
                "test_top5": metrics.get("test_top5", ""),
                "test_loss": metrics.get("test_loss", ""),
                "learning_rate": metrics["epoch_metrics"]["learning_rate"],
            }
            with open(self.metrics_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_metrics.keys()).writerow(csv_metrics)

        if self.use_wandb:
            log_dict = {"epoch": epoch, "training/test-acc": metrics["test_acc"]}
            if "train_acc" in metrics:
                log_dict["training/train-acc"] = metrics["train_acc"]
            if "test_top5" in metrics:
                log_dict["training/test-top5"] = metrics["test_top5"]
            if "train_top5" in metrics:
                log_dict["training/train-top5"] = metrics["train_top5"]
            if "test_loss" in metrics:
                log_dict["training/test-loss"] = metrics["test_loss"]
            try:
                wandb.log(log_dict)
            except Exception as e:
                rprint(f"W&B logging failed: {e}", style="warning")

        status = f"Epoch [{epoch}/{self.cfg.num_epochs}] Test Acc: {metrics['test_acc']:.2f}%"
        if "test_top5" in metrics:
            status += f" (top5: {metrics['test_top5']:.2f}%)"
        if "train_acc" in metrics:
            status += f" Train Acc: {metrics['train_acc']:.2f}%"
            if "train_top5" in metrics:
                status += f" (top5: {metrics['train_top5']:.2f}%)"
        rprint(status, style="info")

    def log_training_step(self, epoch, batch_idx, loss, lr):
        if self.use_wandb:
            try:
                fractional_epoch = epoch - 1 + (batch_idx / self.cfg.train_loader_len)
                wandb.log({"epoch": fractional_epoch, "training/loss": loss, "training/learning_rate": lr})
            except Exception as e:
                rprint(f"W&B step logging failed: {e}", style="warning")

    def finish(self):
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                rprint(f"W&B finish failed: {e}", style="warning")


def setup_optimizer(model, cfg):
    """AdamW/Adam/SGD with zero weight-decay on biases and 1D (norm) params."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    parameters = [
        {"params": decay, "weight_decay": cfg.get("weight_decay", 0.0)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer_name = cfg.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=cfg.learning_rate)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.learning_rate)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def setup_scheduler(optimizer, cfg):
    """LR scheduler with optional linear warmup."""
    scheduler_name = cfg.lr_scheduler.lower()
    warmup_epochs = cfg.get("warmup_epochs", 0)
    total_epochs = cfg.num_epochs
    T_max = total_epochs - warmup_epochs if warmup_epochs > 0 else total_epochs

    if scheduler_name == "steplr":
        # visreps hardcodes step_size=10 regardless of num_epochs, which
        # silently collapses LR to ~0 well before training ends on any run
        # not close to 100 epochs. Scale it off T_max instead.
        main_scheduler = StepLR(optimizer, step_size=max(1, T_max // 3), gamma=0.1)
    elif scheduler_name == "multisteplr":
        default_milestones = [int(T_max * 0.3), int(T_max * 0.6), int(T_max * 0.9)]
        main_scheduler = MultiStepLR(optimizer, milestones=default_milestones, gamma=0.1)
    elif scheduler_name == "cosineannealinglr":
        eta_min = cfg.learning_rate * 0.05
        main_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Invalid LR scheduler name: {cfg.lr_scheduler}")

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    return main_scheduler
