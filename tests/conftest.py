"""
Pytest configuration and fixtures for medical AI assistant tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_medical_text():
    """Sample medical text for testing."""
    return "Patient presents with chest pain and shortness of breath. Diagnosed with pneumonia."


@pytest.fixture
def sample_ner_data():
    """Sample NER data for testing."""
    return [
        {
            "text": "Patient has diabetes and hypertension.",
            "entities": [
                {"text": "diabetes", "label": "PROBLEM", "start": 12, "end": 20},
                {"text": "hypertension", "label": "PROBLEM", "start": 25, "end": 37}
            ]
        },
        {
            "text": "Prescribed metformin for blood sugar control.",
            "entities": [
                {"text": "metformin", "label": "TREATMENT", "start": 11, "end": 20}
            ]
        },
        {
            "text": "CBC and chest X-ray ordered.",
            "entities": [
                {"text": "CBC", "label": "TEST", "start": 0, "end": 3},
                {"text": "chest X-ray", "label": "TEST", "start": 8, "end": 19}
            ]
        }
    ]


@pytest.fixture
def sample_classification_data():
    """Sample classification data for testing."""
    return [
        {
            "text": "Patient admitted with chest pain and elevated troponin.",
            "labels": ["cardiology", "admission_note"]
        },
        {
            "text": "Follow-up visit for diabetes management.",
            "labels": ["endocrinology", "progress_note"]
        },
        {
            "text": "Neurological examination shows peripheral neuropathy.",
            "labels": ["neurology", "consultation"]
        }
    ]


@pytest.fixture
def ner_label_vocab():
    """NER label vocabulary."""
    return {
        "label2id": {
            "O": 0,
            "B-PROBLEM": 1,
            "I-PROBLEM": 2,
            "B-TREATMENT": 3,
            "I-TREATMENT": 4,
            "B-TEST": 5,
            "I-TEST": 6,
            "B-ANATOMY": 7,
            "I-ANATOMY": 8
        },
        "id2label": {
            0: "O",
            1: "B-PROBLEM",
            2: "I-PROBLEM",
            3: "B-TREATMENT",
            4: "I-TREATMENT",
            5: "B-TEST",
            6: "I-TEST",
            7: "B-ANATOMY",
            8: "I-ANATOMY"
        }
    }


@pytest.fixture
def classification_label_vocab():
    """Classification label vocabulary."""
    return {
        "label2id": {
            "cardiology": 0,
            "neurology": 1,
            "endocrinology": 2,
            "admission_note": 3,
            "progress_note": 4,
            "consultation": 5
        },
        "id2label": {
            0: "cardiology",
            1: "neurology",
            2: "endocrinology",
            3: "admission_note",
            4: "progress_note",
            5: "consultation"
        }
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 30522
            self.pad_token_id = 0
        
        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            tokens = text.split()
            max_length = kwargs.get("max_length", 512)
            
            input_ids = [101] + list(range(1, len(tokens) + 1)) + [102]  # CLS + tokens + SEP
            input_ids = input_ids[:max_length]
            
            attention_mask = [1] * len(input_ids)
            
            # Pad to max_length if needed
            if kwargs.get("padding") == "max_length":
                padding_length = max_length - len(input_ids)
                input_ids += [0] * padding_length
                attention_mask += [0] * padding_length
            
            result = {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask])
            }
            
            if kwargs.get("return_offsets_mapping"):
                offsets = [(0, 0)] + [(i, i+1) for i in range(len(tokens))] + [(0, 0)]
                offsets = offsets[:max_length]
                if kwargs.get("padding") == "max_length":
                    offsets += [(0, 0)] * (max_length - len(offsets))
                result["offset_mapping"] = torch.tensor([offsets])
            
            if kwargs.get("return_tensors") == "pt":
                return result
            
            return result
        
        def encode(self, text, **kwargs):
            return self(text, **kwargs)["input_ids"][0].tolist()
        
        def decode(self, token_ids, **kwargs):
            return " ".join([f"token_{i}" for i in token_ids if i not in [0, 101, 102]])
    
    return MockTokenizer()


@pytest.fixture
def sample_documents():
    """Sample medical documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "text": "Diabetes mellitus is a metabolic disorder characterized by high blood sugar.",
            "metadata": {"source": "medical_textbook", "topic": "endocrinology"}
        },
        {
            "id": "doc2",
            "text": "Hypertension is a condition where blood pressure is consistently elevated.",
            "metadata": {"source": "medical_textbook", "topic": "cardiology"}
        },
        {
            "id": "doc3",
            "text": "Metformin is a first-line medication for type 2 diabetes treatment.",
            "metadata": {"source": "drug_database", "topic": "pharmacology"}
        }
    ]


@pytest.fixture
def sample_query():
    """Sample query for RAG testing."""
    return "What is the treatment for diabetes?"


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    class MockEmbedding:
        def encode(self, texts, **kwargs):
            # Return random embeddings
            if isinstance(texts, str):
                texts = [texts]
            np.random.seed(42)
            return np.random.randn(len(texts), 768)
    
    return MockEmbedding()


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import create_app
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API testing."""
    # Mock JWT token (in real tests, generate proper token)
    return {
        "Authorization": "Bearer mock_token_for_testing"
    }


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test_api_key_12345"


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
