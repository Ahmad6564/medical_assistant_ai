"""
Data preparation utilities for medical AI models.
Handles loading, preprocessing, and splitting medical datasets.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """
    Dataset for Named Entity Recognition.
    
    Expected format:
    [
        {
            "text": "Patient has diabetes and hypertension.",
            "entities": [
                {"text": "diabetes", "label": "PROBLEM", "start": 12, "end": 20},
                {"text": "hypertension", "label": "PROBLEM", "start": 25, "end": 37}
            ]
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with tokenization and label alignment."""
        item = self.data[idx]
        text = item["text"]
        entities = item.get("entities", [])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # Create labels (BIO tagging)
        labels = ["O"] * self.max_length
        offset_mapping = encoding["offset_mapping"][0]
        
        for entity in entities:
            start, end = entity["start"], entity["end"]
            label = entity["label"]
            
            # Find tokens that overlap with entity
            entity_tokens = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:  # Special tokens
                    continue
                if token_start >= start and token_end <= end:
                    entity_tokens.append(token_idx)
            
            # Apply BIO tagging
            if entity_tokens:
                labels[entity_tokens[0]] = f"B-{label}"
                for token_idx in entity_tokens[1:]:
                    labels[token_idx] = f"I-{label}"
        
        # Convert labels to IDs
        label_ids = [self.label2id.get(label, self.label2id["O"]) for label in labels]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


class ClassificationDataset(Dataset):
    """
    Dataset for multi-label clinical text classification.
    
    Expected format:
    [
        {
            "text": "Patient admitted for chest pain...",
            "labels": ["cardiology", "progress_note"]
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.num_labels = len(label2id)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with tokenization and multi-label encoding."""
        item = self.data[idx]
        text = item["text"]
        labels = item.get("labels", [])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create multi-label vector
        label_vector = torch.zeros(self.num_labels, dtype=torch.float)
        for label in labels:
            if label in self.label2id:
                label_vector[self.label2id[label]] = 1.0
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vector
        }


def load_ner_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load NER data from JSON file.
    
    Args:
        data_path: Path to JSON file
        
    Returns:
        List of examples with text and entities
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} NER examples from {data_path}")
    return data


def load_classification_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load classification data from JSON file.
    
    Args:
        data_path: Path to JSON file
        
    Returns:
        List of examples with text and labels
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} classification examples from {data_path}")
    return data


def create_ner_label_vocab(data: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label vocabulary for NER (BIO tagging).
    
    Args:
        data: List of NER examples
        
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    # Collect all entity types
    entity_types = set()
    for item in data:
        for entity in item.get("entities", []):
            entity_types.add(entity["label"])
    
    # Create BIO labels
    labels = ["O"]
    for entity_type in sorted(entity_types):
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    logger.info(f"Created label vocabulary with {len(labels)} labels: {labels}")
    return label2id, id2label


def create_classification_label_vocab(
    data: List[Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label vocabulary for classification.
    
    Args:
        data: List of classification examples
        
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    # Collect all labels
    all_labels = set()
    for item in data:
        all_labels.update(item.get("labels", []))
    
    labels = sorted(all_labels)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    logger.info(f"Created label vocabulary with {len(labels)} labels: {labels}")
    return label2id, id2label


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets.
    
    Args:
        data: List of examples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train and temp (val + test)
    train_data, temp_data = train_test_split(
        data,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_size,
        random_state=random_state,
        shuffle=True
    )
    
    logger.info(
        f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
    )
    
    return train_data, val_data, test_data


def create_data_loaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    dataset_class: type,
    tokenizer,
    label2id: Dict[str, int],
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train/val/test sets.
    
    Args:
        train_data: Training examples
        val_data: Validation examples
        test_data: Test examples
        dataset_class: Dataset class to use
        tokenizer: Tokenizer instance
        label2id: Label to ID mapping
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = dataset_class(train_data, tokenizer, label2id, max_length)
    val_dataset = dataset_class(val_data, tokenizer, label2id, max_length)
    test_dataset = dataset_class(test_data, tokenizer, label2id, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(
        f"Created data loaders: "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, "
        f"test={len(test_loader)} batches"
    )
    
    return train_loader, val_loader, test_loader


def analyze_dataset(data: List[Dict[str, Any]], data_type: str = "ner"):
    """
    Analyze and print dataset statistics.
    
    Args:
        data: List of examples
        data_type: Type of data ('ner' or 'classification')
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Analysis ({data_type.upper()})")
    logger.info(f"{'='*60}")
    
    logger.info(f"Total examples: {len(data)}")
    
    # Text length statistics
    text_lengths = [len(item["text"]) for item in data]
    logger.info(f"\nText length statistics:")
    logger.info(f"  Mean: {np.mean(text_lengths):.1f}")
    logger.info(f"  Median: {np.median(text_lengths):.1f}")
    logger.info(f"  Min: {np.min(text_lengths)}")
    logger.info(f"  Max: {np.max(text_lengths)}")
    
    if data_type == "ner":
        # Entity statistics
        entity_counts = Counter()
        for item in data:
            for entity in item.get("entities", []):
                entity_counts[entity["label"]] += 1
        
        logger.info(f"\nEntity type distribution:")
        for label, count in entity_counts.most_common():
            logger.info(f"  {label}: {count}")
    
    elif data_type == "classification":
        # Label statistics
        label_counts = Counter()
        for item in data:
            for label in item.get("labels", []):
                label_counts[label] += 1
        
        logger.info(f"\nLabel distribution:")
        for label, count in label_counts.most_common():
            logger.info(f"  {label}: {count}")
        
        # Multi-label statistics
        labels_per_example = [len(item.get("labels", [])) for item in data]
        logger.info(f"\nLabels per example:")
        logger.info(f"  Mean: {np.mean(labels_per_example):.2f}")
        logger.info(f"  Min: {np.min(labels_per_example)}")
        logger.info(f"  Max: {np.max(labels_per_example)}")
    
    logger.info(f"{'='*60}\n")


def create_synthetic_ner_data(num_examples: int = 100) -> List[Dict[str, Any]]:
    """
    Create synthetic NER data for testing/demo purposes.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        List of synthetic NER examples
    """
    templates = [
        "Patient has {problem} and {problem}.",
        "Prescribed {treatment} for {problem}.",
        "Laboratory tests show {test} results.",
        "Examination revealed {anatomy} abnormality.",
        "Patient reports {problem} and pain in {anatomy}.",
        "Treatment plan includes {treatment} and {treatment}.",
        "Ordered {test} and {test} to evaluate {problem}.",
        "Physical exam: {anatomy} shows signs of {problem}."
    ]
    
    problems = ["diabetes", "hypertension", "pneumonia", "asthma", "arthritis", "infection"]
    treatments = ["metformin", "lisinopril", "albuterol", "antibiotics", "insulin", "aspirin"]
    tests = ["CBC", "chest X-ray", "blood glucose", "ECG", "urinalysis", "CT scan"]
    anatomies = ["heart", "lungs", "abdomen", "chest", "knee", "liver"]
    
    data = []
    np.random.seed(42)
    
    for _ in range(num_examples):
        template = np.random.choice(templates)
        
        # Fill placeholders
        text = template
        entities = []
        offset = 0
        
        while "{" in text:
            start = text.find("{")
            end = text.find("}")
            entity_type = text[start+1:end].upper()
            
            # Select entity text
            if entity_type == "PROBLEM":
                entity_text = np.random.choice(problems)
            elif entity_type == "TREATMENT":
                entity_text = np.random.choice(treatments)
            elif entity_type == "TEST":
                entity_text = np.random.choice(tests)
            else:  # ANATOMY
                entity_text = np.random.choice(anatomies)
            
            # Replace placeholder
            text = text[:start] + entity_text + text[end+1:]
            
            # Add entity
            entities.append({
                "text": entity_text,
                "label": entity_type,
                "start": start,
                "end": start + len(entity_text)
            })
        
        data.append({
            "text": text,
            "entities": entities
        })
    
    logger.info(f"Created {num_examples} synthetic NER examples")
    return data


def create_synthetic_classification_data(num_examples: int = 100) -> List[Dict[str, Any]]:
    """
    Create synthetic classification data for testing/demo purposes.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        List of synthetic classification examples
    """
    templates = {
        ("cardiology", "progress_note"): [
            "Patient with history of heart disease presents with chest pain.",
            "Follow-up visit for cardiac catheterization.",
            "Blood pressure remains elevated despite medication."
        ],
        ("neurology", "consultation"): [
            "Patient referred for evaluation of headaches and dizziness.",
            "Neurological examination shows signs of peripheral neuropathy.",
            "Recommend MRI brain to rule out structural abnormalities."
        ],
        ("oncology", "discharge_summary"): [
            "Patient discharged after chemotherapy cycle.",
            "Tumor markers show improvement after treatment.",
            "Follow-up scheduled in oncology clinic."
        ],
        ("general_medicine", "admission_note"): [
            "Patient admitted with fever and cough.",
            "Chief complaint: abdominal pain and nausea.",
            "Vital signs stable on admission."
        ]
    }
    
    data = []
    np.random.seed(42)
    
    # Convert templates to list for easier random selection
    template_list = list(templates.items())
    
    for _ in range(num_examples):
        # Randomly select a category and its texts
        labels, texts = template_list[np.random.randint(0, len(template_list))]
        # Randomly select a text from that category
        text = texts[np.random.randint(0, len(texts))]
        
        data.append({
            "text": text,
            "labels": list(labels)
        })
    
    logger.info(f"Created {num_examples} synthetic classification examples")
    return data
