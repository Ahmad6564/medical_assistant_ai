"""
Model utilities for Medical AI Assistant.
Provides functions for model loading, saving, and inference.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import pickle


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_model(
    model: torch.nn.Module,
    save_dir: Path,
    model_name: str = "model",
    tokenizer: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model, tokenizer, and configuration.
    
    Args:
        model: PyTorch model to save
        save_dir: Directory to save model
        model_name: Name of the model file
        tokenizer: Optional tokenizer to save
        config: Optional configuration dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if hasattr(model, 'save_pretrained'):
        # HuggingFace model
        model.save_pretrained(save_dir)
    else:
        # PyTorch model
        torch.save(model.state_dict(), save_dir / f"{model_name}.pt")
    
    # Save tokenizer
    if tokenizer is not None:
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(save_dir)
        else:
            with open(save_dir / "tokenizer.pkl", 'wb') as f:
                pickle.dump(tokenizer, f)
    
    # Save config
    if config is not None:
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)


def load_model(
    model_path: Path,
    model_class: Optional[type] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load model from disk.
    
    Args:
        model_path: Path to model directory or file
        model_class: Model class for custom models
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = get_device()
    
    model_path = Path(model_path)
    
    # Try loading as HuggingFace model
    if (model_path / "config.json").exists():
        try:
            model = AutoModel.from_pretrained(model_path)
            model.to(device)
            return model
        except Exception:
            pass
    
    # Try loading as PyTorch state dict
    if model_path.is_file() and model_path.suffix == ".pt":
        if model_class is None:
            raise ValueError("model_class must be provided for .pt files")
        
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
    
    raise ValueError(f"Could not load model from {model_path}")


def load_tokenizer(tokenizer_path: Path) -> Any:
    """
    Load tokenizer from disk.
    
    Args:
        tokenizer_path: Path to tokenizer directory or file
        
    Returns:
        Loaded tokenizer
    """
    tokenizer_path = Path(tokenizer_path)
    
    # Try loading as HuggingFace tokenizer
    if (tokenizer_path / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Try loading as pickle file
    pkl_path = tokenizer_path / "tokenizer.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    raise ValueError(f"Could not load tokenizer from {tokenizer_path}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def freeze_parameters(model: torch.nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze all model parameters.
    
    Args:
        model: PyTorch model
        freeze: If True, freeze parameters; if False, unfreeze
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def freeze_layers(
    model: torch.nn.Module,
    layer_names: list,
    freeze: bool = True
) -> None:
    """
    Freeze or unfreeze specific layers.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze/unfreeze
        freeze: If True, freeze parameters; if False, unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = not freeze


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move batch tensors to device.
    
    Args:
        batch: Dictionary of tensors
        device: Target device
        
    Returns:
        Dictionary with tensors moved to device
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
