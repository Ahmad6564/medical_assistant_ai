"""
Training script for Clinical Classification model.
Multi-label classification with focal loss support.
"""

import argparse
import logging
import sys
from pathlib import Path
from transformers import AdamW

from torch.optim import AdamW


import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classification import ClinicalClassifier
from src.utils.config_loader import ConfigLoader
from scripts.train_utils import (
    BaseTrainer, TrainingMetrics, set_seed, count_parameters
)
from scripts.prepare_data import (
    load_classification_data, create_classification_label_vocab, split_data,
    create_data_loaders, ClassificationDataset, analyze_dataset,
    create_synthetic_classification_data
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassifierTrainer(BaseTrainer):
    """Trainer for clinical classification model."""
    
    def __init__(self, model, optimizer, device, config, id2label):
        super().__init__(model, optimizer, device, config)
        self.id2label = id2label
        self.num_labels = len(id2label)
        
        # Use model's built-in loss (focal loss or BCE)
        self.use_model_loss = True
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels=labels)
            
            # Get loss and logits
            if isinstance(outputs, dict):
                loss = outputs["loss"]
                logits = outputs["logits"]
            else:
                loss = outputs[0]
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate scheduler (for warmup)
            if self.scheduler is not None and self.config.get("scheduler") == "warmup":
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Get predictions (threshold at 0.5)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_labels, all_predictions)
        metrics["loss"] = avg_loss
        
        return metrics
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels=labels)
                
                # Get loss and logits
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                else:
                    loss = outputs[0]
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = (torch.sigmoid(logits) > 0.5).float()
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_labels, all_predictions)
        metrics["loss"] = avg_loss
        
        # Print detailed metrics
        logger.info("\nValidation Metrics:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Micro F1: {metrics['f1_micro']:.4f}")
        logger.info(f"  Macro F1: {metrics['f1_macro']:.4f}")
        logger.info(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        logger.info(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        
        # Per-label metrics
        logger.info("\nPer-label metrics:")
        for idx, label_name in self.id2label.items():
            label_preds = all_predictions[:, idx]
            label_true = all_labels[:, idx]
            if label_true.sum() > 0:  # Only if label appears in validation set
                f1 = f1_score(label_true, label_preds, zero_division=0)
                logger.info(f"  {label_name}: F1={f1:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate multi-label classification metrics."""
        return {
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "subset_accuracy": accuracy_score(y_true, y_pred)
        }


def train_classifier_model(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("Training Clinical Classifier")
    logger.info("="*60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config_loader = ConfigLoader()
    classifier_config = config_loader.load_classifier_config()
    
    # Override config with args
    if args.model_name:
        classifier_config.model_name = args.model_name
    if args.batch_size:
        classifier_config.batch_size = args.batch_size
    if args.num_epochs:
        classifier_config.num_epochs = args.num_epochs
    if args.learning_rate:
        classifier_config.learning_rate = args.learning_rate
    
    logger.info(f"Model name: {classifier_config.model_name}")
    logger.info(f"Use focal loss: {classifier_config.use_focal_loss}")
    
    # Load or create data
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading data from {args.data_path}")
        data = load_classification_data(args.data_path)
    else:
        logger.info("Creating synthetic data for demo")
        data = create_synthetic_classification_data(num_examples=500)
    
    # Analyze dataset
    analyze_dataset(data, data_type="classification")
    
    # Create label vocabulary
    label2id, id2label = create_classification_label_vocab(data)
    num_labels = len(label2id)
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Labels: {list(label2id.keys())}")
    
    # Update config with actual number of labels
    classifier_config.num_labels = num_labels
    
    # Split data
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=args.seed
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(classifier_config.model_name)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        ClassificationDataset,
        tokenizer,
        label2id,
        batch_size=classifier_config.batch_size,
        max_length=classifier_config.max_length,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating clinical classifier model...")
    model = ClinicalClassifier(
        model_name=classifier_config.model_name,
        num_labels=num_labels,
        hidden_dim=classifier_config.hidden_dim,
        dropout=classifier_config.dropout,
        use_focal_loss=classifier_config.use_focal_loss,
        focal_alpha=classifier_config.focal_alpha,
        focal_gamma=classifier_config.focal_gamma
    )
    
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=classifier_config.learning_rate,
        weight_decay=classifier_config.weight_decay
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * classifier_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=classifier_config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(config_loader.load_mlops_config().mlflow_tracking_uri)
        mlflow.set_experiment("classification_training")
        mlflow.start_run(run_name=f"classifier_{args.run_name}")
        
        # Log parameters
        mlflow.log_params({
            "model_name": classifier_config.model_name,
            "batch_size": classifier_config.batch_size,
            "learning_rate": classifier_config.learning_rate,
            "num_epochs": classifier_config.num_epochs,
            "max_length": classifier_config.max_length,
            "num_labels": num_labels,
            "use_focal_loss": classifier_config.use_focal_loss,
            "focal_alpha": classifier_config.focal_alpha,
            "focal_gamma": classifier_config.focal_gamma,
            "dropout": classifier_config.dropout,
            "seed": args.seed
        })
    
    # Training config
    training_config = {
        "num_epochs": classifier_config.num_epochs,
        "save_dir": args.save_dir,
        "early_stopping_patience": 5,
        "early_stopping_delta": 0.001,
        "early_stopping_mode": "max",  # Maximize F1
        "monitor": "val_f1_macro",
        "checkpoint_mode": "max",
        "save_best_only": True,
        "use_mlflow": args.use_mlflow and MLFLOW_AVAILABLE,
        "scheduler": "warmup"
    }
    
    # Create trainer
    trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=training_config,
        id2label=id2label
    )
    trainer.scheduler = scheduler
    
    # Train
    logger.info("Starting training...")
    results = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("Evaluating on test set...")
    logger.info("="*60)
    test_results = trainer.validate_epoch(test_loader)
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_results['loss']:.4f}")
    logger.info(f"  Micro F1: {test_results['f1_micro']:.4f}")
    logger.info(f"  Macro F1: {test_results['f1_macro']:.4f}")
    logger.info(f"  Hamming Loss: {test_results['hamming_loss']:.4f}")
    logger.info(f"  Subset Accuracy: {test_results['subset_accuracy']:.4f}")
    
    # Log final metrics to MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "test_loss": test_results['loss'],
            "test_f1_micro": test_results['f1_micro'],
            "test_f1_macro": test_results['f1_macro'],
            "test_hamming_loss": test_results['hamming_loss'],
            "test_subset_accuracy": test_results['subset_accuracy']
        })
        
        # Save model
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
    
    # Save label vocabulary
    import json
    labels_path = Path(args.save_dir) / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    logger.info(f"Saved label vocabulary to {labels_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train clinical classifier")
    
    # Data arguments
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default=None,
                        help="Path to training data (JSON file)")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", type=str, default="models/classifier",
                        help="Directory to save model checkpoints")
    
    # Model arguments
    parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default=None,
                        help="Pre-trained model name")
    
    # Training arguments
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # MLflow arguments
    parser.add_argument("--use-mlflow", "--use_mlflow", dest="use_mlflow", action="store_true",
                        help="Use MLflow for experiment tracking")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="default",
                        help="MLflow run name")
    
    args = parser.parse_args()
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Train model
    train_classifier_model(args)


if __name__ == "__main__":
    main()
