"""
Simple helper script to generate synthetic data and prepare it for training.
Run this script to quickly set up your training data!
"""

import json
from pathlib import Path
from scripts.prepare_data import (
    create_synthetic_ner_data,
    create_synthetic_classification_data,
    load_ner_data,
    load_classification_data,
    split_data,
    create_ner_label_vocab,
    create_classification_label_vocab
)


def main():
    """Generate synthetic data and prepare for training."""
    print("=" * 60)
    print("Medical AI Assistant - Data Setup")
    print("=" * 60)
    
    # Step 1: Create directories
    print("\n[1/4] Creating directories...")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    # Step 2: Generate synthetic data
    print("\n[2/4] Generating synthetic data...")
    print("  - Creating 500 NER examples...")
    ner_data = create_synthetic_ner_data(num_examples=500)
    
    print("  - Creating 500 classification examples...")
    clf_data = create_synthetic_classification_data(num_examples=500)
    
    # Save raw data
    with open("data/raw/ner_data.json", "w") as f:
        json.dump(ner_data, f, indent=2)
    
    with open("data/raw/classification_data.json", "w") as f:
        json.dump(clf_data, f, indent=2)
    
    print("✓ Synthetic data generated and saved")
    
    # Step 3: Split data
    print("\n[3/4] Splitting data into train/val/test...")
    ner_train, ner_val, ner_test = split_data(ner_data, train_ratio=0.8, val_ratio=0.1)
    clf_train, clf_val, clf_test = split_data(clf_data, train_ratio=0.8, val_ratio=0.1)
    
    # Save split data
    with open("data/processed/ner_train.json", "w") as f:
        json.dump(ner_train, f, indent=2)
    with open("data/processed/ner_val.json", "w") as f:
        json.dump(ner_val, f, indent=2)
    with open("data/processed/ner_test.json", "w") as f:
        json.dump(ner_test, f, indent=2)
    
    with open("data/processed/clf_train.json", "w") as f:
        json.dump(clf_train, f, indent=2)
    with open("data/processed/clf_val.json", "w") as f:
        json.dump(clf_val, f, indent=2)
    with open("data/processed/clf_test.json", "w") as f:
        json.dump(clf_test, f, indent=2)
    
    print(f"✓ Data split: {len(ner_train)} train, {len(ner_val)} val, {len(ner_test)} test")
    
    # Step 4: Create label vocabularies
    print("\n[4/4] Creating label vocabularies...")
    ner_label2id, ner_id2label = create_ner_label_vocab(ner_data)
    clf_label2id, clf_id2label = create_classification_label_vocab(clf_data)
    
    # Create vocabulary dictionaries
    ner_vocab = {
        "label2id": ner_label2id,
        "id2label": ner_id2label
    }
    clf_vocab = {
        "label2id": clf_label2id,
        "id2label": clf_id2label
    }
    
    with open("data/processed/ner_labels.json", "w") as f:
        json.dump(ner_vocab, f, indent=2)
    
    with open("data/processed/clf_labels.json", "w") as f:
        json.dump(clf_vocab, f, indent=2)
    
    print(f"✓ Created NER vocabulary: {len(ner_vocab['label2id'])} labels")
    print(f"✓ Created Classification vocabulary: {len(clf_vocab['label2id'])} labels")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Data Setup Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  data/raw/")
    print("    ├── ner_data.json (500 examples)")
    print("    └── classification_data.json (500 examples)")
    print("  data/processed/")
    print("    ├── ner_train.json")
    print("    ├── ner_val.json")
    print("    ├── ner_test.json")
    print("    ├── clf_train.json")
    print("    ├── clf_val.json")
    print("    ├── clf_test.json")
    print("    ├── ner_labels.json")
    print("    └── clf_labels.json")
    
    print("\n📚 Next steps:")
    print("  1. Train NER model:")
    print("     python scripts/train_ner.py --data-path data/processed \\")
    print("       --save-dir models/ner --use-mlflow --run-name my_experiment")
    print("\n  2. Train classifier:")
    print("     python scripts/train_classifier.py --data-path data/processed \\")
    print("       --save-dir models/classifier --use-mlflow --run-name my_experiment")
    print("\n  3. View results in MLflow UI:")
    print("     python scripts/setup_mlflow.py --start-ui")
    print("     Then open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
