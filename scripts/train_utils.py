"""
Training utilities for medical AI models.
Includes early stopping, checkpointing, metrics tracking, and base trainer classes.
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/F1
        verbose: Print messages
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == "min":
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"Validation score improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best: {self.best_score:.4f} at epoch {self.best_epoch})"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered. Best score: {self.best_score:.4f} "
                        f"at epoch {self.best_epoch}"
                    )
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor ('val_loss', 'val_f1', etc.)
        mode: 'min' for loss, 'max' for accuracy/F1
        save_best_only: Only save when monitored metric improves
        verbose: Print messages
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_score = None
        if mode == "min":
            self.monitor_op = lambda x, y: x < y
        else:
            self.monitor_op = lambda x, y: x > y
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: TrainingMetrics,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save last checkpoint
        last_path = self.save_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            if self.verbose:
                logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        # Save epoch checkpoint
        epoch_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)
    
    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: TrainingMetrics
    ):
        """
        Check if checkpoint should be saved and save it.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metrics: Training metrics
        """
        # Extract monitored metric
        metrics_dict = metrics.to_dict()
        if self.monitor in metrics_dict:
            current_score = metrics_dict[self.monitor]
        elif metrics.val_metrics and self.monitor in metrics.val_metrics:
            current_score = metrics.val_metrics[self.monitor]
        else:
            logger.warning(f"Monitored metric '{self.monitor}' not found")
            return
        
        # Check if this is the best model
        is_best = False
        if self.best_score is None:
            is_best = True
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score):
            is_best = True
            self.best_score = current_score
        
        # Save checkpoint
        if not self.save_best_only or is_best:
            self.save_checkpoint(model, optimizer, metrics.epoch, metrics, is_best)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Path to checkpoint (if None, loads best)
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / "best_checkpoint.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self, save_dir: Optional[str] = None, use_mlflow: bool = False):
        self.save_dir = Path(save_dir) if save_dir else None
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.history: List[TrainingMetrics] = []
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """
        Log metrics for current epoch.
        
        Args:
            metrics: Training metrics to log
        """
        self.history.append(metrics)
        
        # Print to console
        log_str = f"Epoch {metrics.epoch}: "
        log_str += f"train_loss={metrics.train_loss:.4f}"
        if metrics.val_loss is not None:
            log_str += f", val_loss={metrics.val_loss:.4f}"
        if metrics.train_metrics:
            for k, v in metrics.train_metrics.items():
                log_str += f", train_{k}={v:.4f}"
        if metrics.val_metrics:
            for k, v in metrics.val_metrics.items():
                log_str += f", val_{k}={v:.4f}"
        log_str += f", lr={metrics.learning_rate:.2e}"
        log_str += f", time={metrics.epoch_time:.2f}s"
        
        logger.info(log_str)
        
        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_metrics({
                "train_loss": metrics.train_loss,
                "learning_rate": metrics.learning_rate,
                "epoch_time": metrics.epoch_time
            }, step=metrics.epoch)
            
            if metrics.val_loss is not None:
                mlflow.log_metric("val_loss", metrics.val_loss, step=metrics.epoch)
            
            if metrics.train_metrics:
                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in metrics.train_metrics.items()},
                    step=metrics.epoch
                )
            
            if metrics.val_metrics:
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in metrics.val_metrics.items()},
                    step=metrics.epoch
                )
    
    def save_history(self, filename: str = "training_history.json"):
        """Save training history to JSON file."""
        if self.save_dir is None:
            logger.warning("No save directory specified")
            return
        
        history_path = self.save_dir / filename
        history_data = [m.to_dict() for m in self.history]
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Saved training history to {history_path}")
    
    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min") -> int:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric to evaluate
            mode: 'min' or 'max'
            
        Returns:
            Best epoch number
        """
        if not self.history:
            return 0
        
        scores = []
        for m in self.history:
            if metric == "val_loss":
                scores.append(m.val_loss if m.val_loss is not None else float('inf'))
            elif metric == "train_loss":
                scores.append(m.train_loss)
            elif m.val_metrics and metric in m.val_metrics:
                scores.append(m.val_metrics[metric])
            elif m.train_metrics and metric in m.train_metrics:
                scores.append(m.train_metrics[metric])
            else:
                scores.append(float('inf') if mode == "min" else float('-inf'))
        
        if mode == "min":
            best_idx = np.argmin(scores)
        else:
            best_idx = np.argmax(scores)
        
        return self.history[best_idx].epoch


class BaseTrainer:
    """
    Base trainer class for all models.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        device: Device to train on
        config: Training configuration
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Training components
        self.early_stopping = None
        self.checkpoint = None
        self.metrics_tracker = None
        self.scheduler = None
        
        # Setup training components
        self._setup_training()
    
    def _setup_training(self):
        """Setup training components (early stopping, checkpointing, etc.)."""
        # Early stopping
        if self.config.get("early_stopping_patience"):
            self.early_stopping = EarlyStopping(
                patience=self.config["early_stopping_patience"],
                min_delta=self.config.get("early_stopping_delta", 0.0),
                mode=self.config.get("early_stopping_mode", "min"),
                verbose=True
            )
        
        # Checkpointing
        if self.config.get("save_dir"):
            self.checkpoint = ModelCheckpoint(
                save_dir=self.config["save_dir"],
                monitor=self.config.get("monitor", "val_loss"),
                mode=self.config.get("checkpoint_mode", "min"),
                save_best_only=self.config.get("save_best_only", True),
                verbose=True
            )
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(
            save_dir=self.config.get("save_dir"),
            use_mlflow=self.config.get("use_mlflow", False)
        )
        
        # Learning rate scheduler
        scheduler_type = self.config.get("scheduler")
        if scheduler_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=True
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"]
            )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch. To be implemented by subclasses."""
        raise NotImplementedError
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch. To be implemented by subclasses."""
        raise NotImplementedError
    
    def train(self, train_loader, val_loader=None) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training results dictionary
        """
        num_epochs = self.config["num_epochs"]
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_results = self.train_epoch(train_loader)
            
            # Validate
            val_results = None
            if val_loader is not None:
                val_results = self.validate_epoch(val_loader)
            
            # Create metrics object
            epoch_time = time.time() - epoch_start
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_results["loss"],
                val_loss=val_results["loss"] if val_results else None,
                train_metrics={k: v for k, v in train_results.items() if k != "loss"},
                val_metrics={k: v for k, v in val_results.items() if k != "loss"} if val_results else None,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                epoch_time=epoch_time
            )
            
            # Log metrics
            self.metrics_tracker.log_metrics(metrics)
            
            # Save checkpoint
            if self.checkpoint:
                self.checkpoint(self.model, self.optimizer, metrics)
            
            # Update learning rate (skip if warmup scheduler - already stepped in train_epoch)
            if self.scheduler and self.config.get("scheduler") != "warmup":
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics.val_loss if metrics.val_loss else metrics.train_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping and val_results:
                if self.early_stopping(val_results["loss"], epoch):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Save training history
        self.metrics_tracker.save_history()
        
        # Get best epoch
        best_epoch = self.metrics_tracker.get_best_epoch()
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        
        return {
            "best_epoch": best_epoch,
            "history": self.metrics_tracker.history
        }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0
