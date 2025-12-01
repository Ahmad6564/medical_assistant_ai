"""
Unit tests for clinical classification model.
"""

import pytest
import torch
import numpy as np
from src.models.classification import ClinicalClassifier, FocalLoss, AsymmetricLoss


@pytest.mark.unit
class TestClinicalClassifier:
    """Tests for ClinicalClassifier model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = ClinicalClassifier(
            model_name="bert-base-uncased",
            num_labels=8,
            dropout=0.1,
            use_focal_loss=True
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, device):
        """Test forward pass works."""
        model = ClinicalClassifier(
            model_name="bert-base-uncased",
            num_labels=8,
            use_focal_loss=False
        ).to(device)
        
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        
        assert logits.shape == (batch_size, 8)
    
    def test_with_labels(self, device):
        """Test model with labels for training."""
        model = ClinicalClassifier(
            model_name="bert-base-uncased",
            num_labels=8,
            use_focal_loss=True
        ).to(device)
        
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)
        labels = torch.zeros(batch_size, 8).to(device)
        labels[:, :3] = 1.0  # Set first 3 labels as positive
        
        model.train()
        outputs = model(input_ids, attention_mask, token_type_ids, labels)
        
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"].item() > 0
    
    def test_prediction(self, device):
        """Test prediction with sigmoid activation."""
        model = ClinicalClassifier(
            model_name="bert-base-uncased",
            num_labels=8
        ).to(device)
        model.eval()
        
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = torch.sigmoid(logits)
        
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


@pytest.mark.unit
class TestFocalLoss:
    """Tests for Focal Loss."""
    
    def test_initialization(self):
        """Test FocalLoss can be initialized."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        assert loss_fn is not None
    
    def test_forward_pass(self, device):
        """Test loss calculation."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        batch_size = 4
        num_classes = 8
        
        logits = torch.randn(batch_size, num_classes).to(device)
        targets = torch.zeros(batch_size, num_classes).to(device)
        targets[:, 0] = 1.0
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_reduction_modes(self, device):
        """Test different reduction modes."""
        batch_size = 4
        num_classes = 8
        
        logits = torch.randn(batch_size, num_classes).to(device)
        targets = torch.zeros(batch_size, num_classes).to(device)
        
        loss_mean = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")(logits, targets)
        loss_sum = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")(logits, targets)
        loss_none = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")(logits, targets)
        
        assert loss_mean.numel() == 1
        assert loss_sum.numel() == 1
        assert loss_none.shape == (batch_size, num_classes)


@pytest.mark.unit
class TestAsymmetricLoss:
    """Tests for Asymmetric Loss."""
    
    def test_initialization(self):
        """Test AsymmetricLoss can be initialized."""
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1)
        assert loss_fn is not None
    
    def test_forward_pass(self, device):
        """Test loss calculation."""
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1)
        
        batch_size = 4
        num_classes = 8
        
        logits = torch.randn(batch_size, num_classes).to(device)
        targets = torch.zeros(batch_size, num_classes).to(device)
        targets[:, :2] = 1.0
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


@pytest.mark.unit
def test_classification_data_format(sample_classification_data):
    """Test classification data format."""
    assert len(sample_classification_data) == 3
    assert "text" in sample_classification_data[0]
    assert "labels" in sample_classification_data[0]
    assert isinstance(sample_classification_data[0]["labels"], list)


@pytest.mark.unit
def test_classification_label_vocab(classification_label_vocab):
    """Test classification label vocabulary."""
    assert "label2id" in classification_label_vocab
    assert "id2label" in classification_label_vocab
    assert "cardiology" in classification_label_vocab["label2id"]
    assert len(classification_label_vocab["label2id"]) == 6


@pytest.mark.integration
@pytest.mark.slow
def test_classifier_training_step(device, sample_classification_data, mock_tokenizer, classification_label_vocab):
    """Test one training step of classifier."""
    from scripts.prepare_data import ClassificationDataset
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = ClassificationDataset(
        sample_classification_data,
        mock_tokenizer,
        classification_label_vocab["label2id"],
        max_length=128
    )
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=2)
    
    # Create model
    model = ClinicalClassifier(
        model_name="bert-base-uncased",
        num_labels=len(classification_label_vocab["label2id"]),
        use_focal_loss=True
    ).to(device)
    
    # Get batch
    batch = next(iter(loader))
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
    labels = batch["labels"].to(device)
    
    # Forward pass
    model.train()
    outputs = model(input_ids, attention_mask, token_type_ids, labels)
    
    # Check outputs
    assert "loss" in outputs
    assert "logits" in outputs
    
    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    outputs["loss"].backward()
    optimizer.step()


@pytest.mark.unit
def test_multi_label_metrics():
    """Test multi-label classification metrics."""
    from sklearn.metrics import f1_score, hamming_loss
    
    y_true = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    y_pred = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    hamming = hamming_loss(y_true, y_pred)
    
    assert 0.0 <= f1_micro <= 1.0
    assert 0.0 <= f1_macro <= 1.0
    assert 0.0 <= hamming <= 1.0
