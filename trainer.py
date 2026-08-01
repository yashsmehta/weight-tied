"""Training loop, config-driven. Ported from visreps/trainer.py with
visreps/test_loss_tracking.patch folded in (evaluate() returns a 3-tuple
including average loss on the split, not just top1/top5 accuracy).
"""
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import model_utils
import utils
from dataloaders.imagenet import get_obj_cls_loader
from utils import MetricsLogger, calculate_cls_accuracy, is_interactive_environment


class Trainer:
    """Trainer for object classification models (ECTiedNet, AlexNet)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()

    def _setup(self):
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.datasets, self.loaders = get_obj_cls_loader(self.cfg)
        num_classes = self.datasets["train"].num_classes
        self.model = model_utils.load_model(self.cfg, self.device, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = utils.setup_optimizer(self.model, self.cfg)
        self.scheduler = utils.setup_scheduler(self.optimizer, self.cfg)

        self.checkpoint_dir = None
        self.cfg_dict = None
        if self.cfg.log_checkpoints:
            self.checkpoint_dir, self.cfg_dict = model_utils.setup_checkpoint_dir(self.cfg, self.model)
            model_utils.save_checkpoint(self.checkpoint_dir, 0, self.model, self.optimizer, {}, self.cfg_dict)

        self.metrics_logger = MetricsLogger(self.cfg, self.checkpoint_dir)

    @torch.no_grad()
    def evaluate(self, split="test"):
        self.model.eval()
        return calculate_cls_accuracy(self.loaders[split], self.model, self.device, criterion=self.criterion)

    def _compute_gradients(self, loss):
        """Compute gradients and optionally clip them."""
        loss.backward()
        if hasattr(self.cfg, "grad_clip") and self.cfg.grad_clip > 0:
            return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        return torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None]))

    def _create_progress_bar(self, loader, epoch):
        if is_interactive_environment():
            return tqdm(loader, desc=f"Epoch {epoch}", leave=False), True
        print(f"Starting Epoch {epoch}")
        return loader, False

    def train_epoch(self, epoch):
        self.model.train()
        loader, use_pbar = self._create_progress_bar(self.loaders["train"], epoch)

        total_loss = 0
        total_grad_norm = 0

        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)

            grad_norm = self._compute_gradients(loss)
            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            lr = self.optimizer.param_groups[0]["lr"]

            self.metrics_logger.log_training_step(epoch, i, loss.item(), lr)

            if use_pbar:
                avg_loss = total_loss / (i + 1)
                loader.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "LR": f"{lr:.6f}", "Grad Norm": f"{grad_norm:.4f}"})

        avg_loss = total_loss / len(loader)
        return avg_loss, {"epoch_loss": avg_loss, "learning_rate": lr}

    def train(self):
        start_time = time.time()
        for epoch in range(1, self.cfg.num_epochs + 1):
            epoch_loss, epoch_metrics = self.train_epoch(epoch)
            self.scheduler.step()

            metrics = {"epoch": epoch, "epoch_metrics": epoch_metrics}

            if epoch == 1 and is_interactive_environment():
                eta_seconds = (time.time() - start_time) * (self.cfg.num_epochs - 1)
                hours, minutes = int(eta_seconds // 3600), int((eta_seconds % 3600) // 60)
                time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                print(f"Estimated time remaining: {time_str}")

            if epoch % self.cfg.log_interval == 0:
                for split in ["test", "train"]:
                    top1, top5, split_loss = self.evaluate(split)
                    metrics[f"{split}_acc"] = top1
                    metrics[f"{split}_top5"] = top5
                    if split == "test":
                        metrics["test_loss"] = split_loss

                self.metrics_logger.log_metrics(epoch, epoch_loss, metrics)

            if self.cfg.log_checkpoints and epoch % self.cfg.checkpoint_interval == 0:
                model_utils.save_checkpoint(
                    self.checkpoint_dir, epoch, self.model, self.optimizer, metrics, self.cfg_dict
                )

        self.metrics_logger.finish()
        return self.model
