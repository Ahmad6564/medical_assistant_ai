"""
Training script for Named Entity Recognition models.
Supports both BiLSTM-CRF and Transformer-based models.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.optim import AdamW

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ner import TransformerNER, BiLSTM_CRF, MedicalNER
from src.utils.config_loader import ConfigLoader
from scripts.train_utils import (
    BaseTrainer, TrainingMetrics, set_seed, count_parameters
)
from scripts.prepare_data import (
    load_ner_data, create_ner_label_vocab, split_data,
    create_data_loaders, NERDataset, analyze_dataset,
    create_synthetic_ner_data
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


class NERTrainer(BaseTrainer):
    """Trainer for NER models."""
    
    def __init__(self, model, optimizer, device, config, id2label):
        super().__init__(model, optimizer, device, config)
        self.id2label = id2label
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
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
            
            if hasattr(self.model, 'forward'):
                outputs = self.model(input_ids, attention_mask, labels=labels)
            else:
                outputs = self.model(input_ids, attention_mask, labels=labels)
            
            # Calculate loss
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
                logits = outputs["logits"]
            else:
                logits = outputs
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate scheduler (for warmup)
            if self.scheduler is not None and self.config.get("scheduler") == "warmup":
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to labels for seqeval
            for pred, label, mask in zip(predictions, labels, attention_mask):
                pred_labels = []
                true_labels = []
                for p, l, m in zip(pred, label, mask):
                    if m == 1 and l != -100:
                        pred_labels.append(self.id2label.get(p.item(), "O"))
                        true_labels.append(self.id2label.get(l.item(), "O"))
                
                if pred_labels:
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        return {
            "loss": avg_loss,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
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
                if hasattr(self.model, 'forward'):
                    outputs = self.model(input_ids, attention_mask, labels=labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels=labels)
                
                # Calculate loss
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                else:
                    logits = outputs
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Convert to labels for seqeval
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    pred_labels = []
                    true_labels = []
                    for p, l, m in zip(pred, label, mask):
                        if m == 1 and l != -100:
                            pred_labels.append(self.id2label.get(p.item(), "O"))
                            true_labels.append(self.id2label.get(l.item(), "O"))
                    
                    if pred_labels:
                        all_predictions.append(pred_labels)
                        all_labels.append(true_labels)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        # Print classification report
        logger.info("\nValidation Classification Report:")
        logger.info(classification_report(all_labels, all_predictions))
        
        return {
            "loss": avg_loss,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


def train_ner_model(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("Training NER Model")
    logger.info("="*60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config_loader = ConfigLoader()
    ner_config = config_loader.load_ner_config()
    
    # Override config with args
    if args.model_type:
        ner_config.model_type = args.model_type
    if args.model_name:
        ner_config.model_name = args.model_name
    if args.batch_size:
        ner_config.batch_size = args.batch_size
    if args.num_epochs:
        ner_config.num_epochs = args.num_epochs
    if args.learning_rate:
        ner_config.learning_rate = args.learning_rate
    
    logger.info(f"Model type: {ner_config.model_type}")
    logger.info(f"Model name: {ner_config.model_name}")
    
    # Load or create data
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading data from {args.data_path}")
        data = load_ner_data(args.data_path)
    else:
        logger.info("Creating synthetic data for demo")
        data = create_synthetic_ner_data(num_examples=500)
    
    # Analyze dataset
    analyze_dataset(data, data_type="ner")
    
    # Create label vocabulary
    label2id, id2label = create_ner_label_vocab(data)
    num_labels = len(label2id)
    logger.info(f"Number of labels: {num_labels}")
    
    # Split data
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=args.seed
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ner_config.model_name)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        NERDataset,
        tokenizer,
        label2id,
        batch_size=ner_config.batch_size,
        max_length=ner_config.max_length,
        num_workers=0
    )
    
    # Create model
    logger.info(f"Creating {ner_config.model_type} model...")
    
    if ner_config.model_type == "transformer":
        model = TransformerNER(
            model_name=ner_config.model_name,
            num_labels=num_labels,
            dropout=ner_config.dropout,
            use_crf=ner_config.use_crf
        )
    elif ner_config.model_type == "bilstm_crf":
        # For BiLSTM-CRF, we need vocabulary
        vocab_size = tokenizer.vocab_size
        model = BiLSTM_CRF(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=ner_config.hidden_dim,
            num_tags=num_labels,
            num_layers=ner_config.num_lstm_layers,
            dropout=ner_config.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {ner_config.model_type}")
    
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=ner_config.learning_rate,
        weight_decay=ner_config.weight_decay
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * ner_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=ner_config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(config_loader.load_mlops_config().mlflow_tracking_uri)
        mlflow.set_experiment("ner_training")
        mlflow.start_run(run_name=f"ner_{ner_config.model_type}_{args.run_name}")
        
        # Log parameters
        mlflow.log_params({
            "model_type": ner_config.model_type,
            "model_name": ner_config.model_name,
            "batch_size": ner_config.batch_size,
            "learning_rate": ner_config.learning_rate,
            "num_epochs": ner_config.num_epochs,
            "max_length": ner_config.max_length,
            "num_labels": num_labels,
            "use_crf": ner_config.use_crf,
            "dropout": ner_config.dropout,
            "seed": args.seed
        })
    
    # Training config
    training_config = {
        "num_epochs": ner_config.num_epochs,
        "save_dir": args.save_dir,
        "early_stopping_patience": 5,
        "early_stopping_delta": 0.001,
        "early_stopping_mode": "max",  # Maximize F1
        "monitor": "val_f1",
        "checkpoint_mode": "max",
        "save_best_only": True,
        "use_mlflow": args.use_mlflow and MLFLOW_AVAILABLE,
        "scheduler": "warmup"
    }
    
    # Create trainer
    trainer = NERTrainer(
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
    logger.info(f"  F1: {test_results['f1']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall: {test_results['recall']:.4f}")
    
    # Log final metrics to MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "test_loss": test_results['loss'],
            "test_f1": test_results['f1'],
            "test_precision": test_results['precision'],
            "test_recall": test_results['recall']
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
    parser = argparse.ArgumentParser(description="Train NER model")
    
    # Data arguments
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default=None,
                        help="Path to training data (JSON file)")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", type=str, default="models/ner",
                        help="Directory to save model checkpoints")
    
    # Model arguments
    parser.add_argument("--model-type", "--model_type", dest="model_type", type=str, choices=["transformer", "bilstm_crf"],
                        default=None, help="Model architecture")
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
    train_ner_model(args)


if __name__ == "__main__":
    main()
