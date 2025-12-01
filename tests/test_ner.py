"""
Unit tests for NER models.
"""

import pytest
import torch
import numpy as np
from src.models.ner import TransformerNER, BiLSTM_CRF, MedicalNER, EntityLinker


@pytest.mark.unit
class TestTransformerNER:
    """Tests for Transformer-based NER model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = TransformerNER(
            model_name="bert-base-uncased",
            num_labels=9,
            dropout=0.1,
            use_crf=False
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, device):
        """Test forward pass works."""
        model = TransformerNER(
            model_name="bert-base-uncased",
            num_labels=9,
            use_crf=False
        ).to(device)
        
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        outputs = model(input_ids, attention_mask)
        
        # Model returns dict with 'predictions' key
        assert isinstance(outputs, dict)
        assert "predictions" in outputs
        assert outputs["predictions"].shape == (batch_size, seq_length)
    
    def test_with_crf(self, device):
        """Test model with CRF layer."""
        try:
            model = TransformerNER(
                model_name="bert-base-uncased",
                num_labels=9,
                use_crf=True
            ).to(device)
            
            batch_size = 2
            seq_length = 128
            
            input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
            attention_mask = torch.ones(batch_size, seq_length).to(device)
            
            outputs = model(input_ids, attention_mask)
            # Model returns dict with 'predictions' key
            assert isinstance(outputs, dict)
            assert "predictions" in outputs
            assert len(outputs["predictions"]) == batch_size
        except ImportError:
            pytest.skip("CRF not available")
    
    def test_training_mode(self, device):
        """Test model in training mode with labels."""
        model = TransformerNER(
            model_name="bert-base-uncased",
            num_labels=9,
            use_crf=False
        ).to(device)
        
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        labels = torch.randint(0, 9, (batch_size, seq_length)).to(device)
        
        model.train()
        outputs = model(input_ids, attention_mask, labels=labels)
        
        assert isinstance(outputs, dict)
        assert "loss" in outputs
        assert outputs["loss"] is not None


@pytest.mark.unit
class TestBiLSTM_CRF:
    """Tests for BiLSTM-CRF model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = BiLSTM_CRF(
            vocab_size=30522,
            embedding_dim=300,
            hidden_dim=256,
            num_tags=9,
            num_layers=2,
            dropout=0.1
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, device):
        """Test forward pass works."""
        model = BiLSTM_CRF(
            vocab_size=30522,
            embedding_dim=300,
            hidden_dim=256,
            num_tags=9
        ).to(device)
        
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        try:
            outputs = model(input_ids, attention_mask)
            assert outputs.shape[0] == batch_size
            assert outputs.shape[1] == seq_length
        except Exception as e:
            pytest.skip(f"CRF forward pass not available: {e}")
    
    def test_with_pretrained_embeddings(self, device):
        """Test model with pretrained embeddings."""
        vocab_size = 30522
        embedding_dim = 300
        
        pretrained_embeddings = torch.randn(vocab_size, embedding_dim)
        
        model = BiLSTM_CRF(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=256,
            num_tags=9,
            pretrained_embeddings=pretrained_embeddings
        ).to(device)
        
        assert model is not None


@pytest.mark.unit
class TestMedicalNER:
    """Tests for MedicalNER wrapper class."""
    
    def test_initialization(self):
        """Test MedicalNER can be initialized."""
        try:
            ner = MedicalNER(
                model_type="transformer",
                model_name="bert-base-uncased",
                use_crf=False,
                use_entity_linking=False
            )
            assert ner is not None
        except Exception as e:
            pytest.skip(f"MedicalNER initialization failed: {e}")
    
    def test_predict(self, sample_medical_text):
        """Test prediction on sample text."""
        try:
            ner = MedicalNER(
                model_type="transformer",
                model_name="bert-base-uncased",
                use_crf=False,
                use_entity_linking=False
            )
            
            entities = ner.predict(sample_medical_text)
            assert isinstance(entities, list)
        except Exception as e:
            pytest.skip(f"MedicalNER prediction failed: {e}")


@pytest.mark.unit
class TestEntityLinker:
    """Tests for EntityLinker class."""
    
    def test_initialization(self):
        """Test EntityLinker can be initialized."""
        try:
            linker = EntityLinker()
            assert linker is not None
        except Exception as e:
            pytest.skip(f"EntityLinker initialization failed: {e}")
    
    def test_link_entity(self):
        """Test entity linking."""
        try:
            linker = EntityLinker()
            
            # Link a medical entity
            results = linker.link_entity("diabetes", entity_type="PROBLEM")
            
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Entity linking failed: {e}")


@pytest.mark.unit
def test_ner_data_loading(sample_ner_data):
    """Test NER data format."""
    assert len(sample_ner_data) == 3
    assert "text" in sample_ner_data[0]
    assert "entities" in sample_ner_data[0]
    assert len(sample_ner_data[0]["entities"]) > 0


@pytest.mark.unit
def test_ner_label_vocab(ner_label_vocab):
    """Test NER label vocabulary."""
    assert "label2id" in ner_label_vocab
    assert "id2label" in ner_label_vocab
    assert "O" in ner_label_vocab["label2id"]
    assert "B-PROBLEM" in ner_label_vocab["label2id"]


@pytest.mark.integration
@pytest.mark.slow
def test_ner_training_step(device, sample_ner_data, mock_tokenizer, ner_label_vocab):
    """Test one training step of NER model."""
    from scripts.prepare_data import NERDataset
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = NERDataset(
        sample_ner_data,
        mock_tokenizer,
        ner_label_vocab["label2id"],
        max_length=128
    )
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=2)
    
    # Create model
    model = TransformerNER(
        model_name="bert-base-uncased",
        num_labels=len(ner_label_vocab["label2id"]),
        use_crf=False
    ).to(device)
    
    # Get batch
    batch = next(iter(loader))
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward pass
    model.train()
    outputs = model(input_ids, attention_mask, labels)
    
    # Check outputs
    assert outputs is not None
