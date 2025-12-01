# Complete Setup Guide - Medical AI Assistant
## From Zero to Running Model - Beginner Friendly

This guide will walk you through **everything** from scratch, in the exact order you need to follow.

---

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher installed
- pip (Python package manager)
- Git installed (optional, but recommended)
- At least 4GB of free disk space

Check your Python version:
```bash
python --version
```

---

## 🚀 Step-by-Step Setup (Follow This Exact Order)

### Step 1: Navigate to Project Directory (30 seconds)

Open your terminal/command prompt and navigate to the project:

```bash
cd C:\Users\muhammadahmad5\Desktop\medical-ai-assistant
```

Verify you're in the right place:
```bash
dir
```

You should see folders like `src/`, `scripts/`, `tests/`, etc.

---

### Step 2: Install Required Dependencies (3-5 minutes)

Install all necessary Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for ML models)
- Transformers (for BERT models)
- MLflow (for experiment tracking)
- FastAPI (for API)
- Pytest (for testing)
- And many more...

**If you get an error**, try:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Verify Installation (1 minute)

Test that key packages are installed:

```bash
python -c "import torch; import transformers; import mlflow; print('✓ All packages installed successfully!')"
```

If this prints the success message, you're good to go!

---

### Step 4: Initialize MLflow (2 minutes)

MLflow tracks your experiments and models. Set it up:

```bash
python scripts/setup_mlflow.py
```

You should see output like:
```
Creating MLflow directories...
✓ Created ./mlruns
✓ Created ./mlartifacts
Configuring MLflow tracking...
✓ Tracking URI set to: ./mlruns
✓ MLflow setup completed successfully!
```

**Optional**: Start MLflow UI to view experiments later:
```bash
python scripts/setup_mlflow.py --start-ui --background
```

Then visit: http://localhost:5000

---



python scripts/train_ner.py --data-path data/processed/ner_train.json --save-dir models/ner --use-mlflow --run-name clinicalbert_500examples


## 📊 About the Dataset - IMPORTANT!

### ⚠️ You Don't Need Real Data to Get Started!

The project includes **synthetic data generation** which creates fake medical data for testing. This is perfect for:
- Learning how the system works
- Testing the code
- Developing new features
- Running experiments

### Option A: Use Synthetic Data (Recommended for Beginners) ⭐

This is the **easiest way** to get started immediately!

#### Step 5A: Generate Synthetic Dataset (2 minutes)

Create synthetic medical data for training:

```bash
python -c "from scripts.prepare_data import create_synthetic_ner_data, create_synthetic_classification_data; import json; from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True); ner_data = create_synthetic_ner_data(100); clf_data = create_synthetic_classification_data(100); json.dump(ner_data, open('data/raw/ner_data.json', 'w'), indent=2); json.dump(clf_data, open('data/raw/classification_data.json', 'w'), indent=2); print('✓ Synthetic data created!')"
```

**What this does:**
- Creates 100 synthetic medical text examples for NER
- Creates 100 synthetic examples for classification
- Saves them to `data/raw/` folder

**Verify data was created:**
```bash
dir data\raw
```

You should see `ner_data.json` and `classification_data.json`.

---

### Option B: Use Real Medical Data (Advanced)

If you want to use real medical datasets, here are some options:

#### 1. **n2c2 (formerly i2b2) Datasets** (Medical NLP Standard)
- **Website**: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **What it contains**: De-identified clinical notes
- **Tasks**: NER, classification, temporal relations
- **Access**: Request access (free for research)
- **How to use**: 
  1. Sign up and request dataset access
  2. Download the dataset
  3. Place files in `data/raw/`
  4. Convert to required format (see below)

#### 2. **MIMIC-III Clinical Database**
- **Website**: https://mimic.mit.edu/
- **What it contains**: ICU patient records
- **Access**: Complete CITI training + sign data use agreement
- **Note**: Requires ethics approval for research use

#### 3. **PubMed Abstracts** (Public, Easy to Get)
- **Website**: https://pubmed.ncbi.nlm.nih.gov/
- **What it contains**: Medical research abstracts
- **Access**: Freely available via API
- **How to download**:

```python
# Install biopython first: pip install biopython
from Bio import Entrez
import json

Entrez.email = "your.email@example.com"

# Search for diabetes-related papers
handle = Entrez.esearch(db="pubmed", term="diabetes", retmax=100)
record = Entrez.read(handle)
ids = record["IdList"]

# Fetch abstracts
handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
abstracts = handle.read()

# Save to file
with open('data/raw/pubmed_abstracts.txt', 'w', encoding='utf-8') as f:
    f.write(abstracts)

print(f"✓ Downloaded {len(ids)} abstracts")
```

#### 4. **DDI (Drug-Drug Interaction) Corpus**
- **Website**: https://github.com/isegura/DDICorpus
- **What it contains**: Drug mentions and interactions
- **Access**: Freely available on GitHub
- **How to get**:

```bash
cd data/raw
git clone https://github.com/isegura/DDICorpus.git
```

#### 5. **MedNLI Dataset** (Medical Natural Language Inference)
- **Website**: https://physionet.org/content/mednli/
- **What it contains**: Clinical text pairs for inference
- **Access**: Free with PhysioNet account

---

### Data Format Requirements

If you're using your own data, it needs to be in this format:

**For NER (Named Entity Recognition):**
```json
[
  {
    "text": "Patient diagnosed with diabetes mellitus.",
    "entities": [
      {
        "start": 23,
        "end": 40,
        "label": "DISEASE",
        "text": "diabetes mellitus"
      }
    ]
  }
]
```

**For Classification:**
```json
[
  {
    "text": "Patient presents with chest pain and shortness of breath.",
    "labels": ["cardiology", "emergency"]
  }
]
```

Save your data as:
- `data/raw/ner_data.json` for NER
- `data/raw/classification_data.json` for classification

---

## 🎓 Continue Training (Following Option A - Synthetic Data)

### Step 6: Prepare Training Data (1 minute)

Split the data into train/validation/test sets:

```bash
python -c "from scripts.prepare_data import load_ner_data, load_classification_data, split_data, create_ner_label_vocab, create_classification_label_vocab; import json; from pathlib import Path; ner_data = load_ner_data('data/raw/ner_data.json'); clf_data = load_classification_data('data/raw/classification_data.json'); ner_train, ner_val, ner_test = split_data(ner_data); clf_train, clf_val, clf_test = split_data(clf_data); Path('data/processed').mkdir(parents=True, exist_ok=True); json.dump(ner_train, open('data/processed/ner_train.json', 'w')); json.dump(ner_val, open('data/processed/ner_val.json', 'w')); json.dump(ner_test, open('data/processed/ner_test.json', 'w')); json.dump(clf_train, open('data/processed/clf_train.json', 'w')); json.dump(clf_val, open('data/processed/clf_val.json', 'w')); json.dump(clf_test, open('data/processed/clf_test.json', 'w')); ner_vocab = create_ner_label_vocab(ner_data); clf_vocab = create_classification_label_vocab(clf_data); json.dump(ner_vocab, open('data/processed/ner_labels.json', 'w')); json.dump(clf_vocab, open('data/processed/clf_labels.json', 'w')); print('✓ Data prepared and split!')"
```

**What this does:**
- Splits data: 80% train, 10% validation, 10% test
- Creates label vocabularies
- Saves everything to `data/processed/`

---

### Step 7: Run Tests (Optional but Recommended) (2 minutes)

Verify everything is working:

```bash
pytest tests/ -v -m "not slow"
```

This runs fast tests to ensure the code works. You should see mostly passing tests (some may be skipped if optional dependencies aren't installed - that's OK!).

**If tests fail:**
- Check that you installed all dependencies (Step 2)
- Some tests may require optional packages - you can ignore those
- As long as most tests pass, you're good!

---

### Step 8: Train Your First Model! (5-10 minutes) 🎉

Now for the exciting part - training a model!

#### Train NER Model:

```bash
python scripts/train_ner.py --data-path data/processed --save-dir models/ner --model-type transformer --model-name bert-base-uncased --batch-size 8 --epochs 3 --lr 2e-5 --use-mlflow --run-name my_first_ner_model
```

**What this command does:**
- `--data-path data/processed`: Where to find training data
- `--save-dir models/ner`: Where to save the trained model
- `--model-type transformer`: Use BERT-based model
- `--model-name bert-base-uncased`: Specific BERT model
- `--batch-size 8`: Process 8 examples at once
- `--epochs 3`: Train for 3 complete passes through data
- `--lr 2e-5`: Learning rate (how fast to learn)
- `--use-mlflow`: Track experiment in MLflow
- `--run-name my_first_ner_model`: Name for this experiment

**You'll see output like:**
```
Epoch 1/3: 100%|████████| Loss: 0.543
Validation F1: 0.756
✓ New best model saved!
...
```

**Training time:**
- CPU: 5-10 minutes
- GPU: 2-3 minutes

---

### Step 9: View Training Results (30 seconds)

Open MLflow UI to see your results:

1. **If you started MLflow UI in Step 4:**
   - Go to: http://localhost:5000

2. **If you didn't start it yet:**
   ```bash
   python scripts/setup_mlflow.py --start-ui
   ```
   - Then go to: http://localhost:5000

In the UI, you'll see:
- Your experiment "my_first_ner_model"
- Metrics (F1 score, precision, recall)
- Training curves
- Saved model files

---

### Step 10: Train Classification Model (Optional) (5-10 minutes)

Train a classification model too:

```bash
python scripts/train_classifier.py --data-path data/processed --save-dir models/classifier --model-name bert-base-uncased --batch-size 8 --epochs 3 --lr 2e-5 --use-mlflow --run-name my_first_classifier
```

---

## 🎯 What to Do Next

### Test the API (if you want to use the trained models)

1. **Start the API server:**
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```

2. **Test in browser:**
   - Go to: http://localhost:8000/docs
   - You'll see interactive API documentation
   - Try the `/health` endpoint

3. **Make predictions:**
   Use the `/predict/ner` or `/predict/classify` endpoints

---

### Register Your Model

Save your trained model to the registry:

```bash
python scripts/model_registry.py register --model-path models/classifier --name medical_classifier --description "Clinical note classifier - 8 labels"
```

---

### Monitor Performance

Track how your model performs:

```bash
python scripts/monitor_models.py --model-name medical_ner --report monitoring_report.json
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "pip: command not found"
**Solution:**
```bash
python -m pip install -r requirements.txt
```

### Issue 2: "CUDA out of memory" during training
**Solution:** Reduce batch size:
```bash
python scripts/train_ner.py ... --batch-size 4
```

### Issue 3: "No module named 'transformers'"
**Solution:** Install missing package:
```bash
pip install transformers
```

### Issue 4: MLflow UI won't start
**Solution:** Use a different port:
```bash
mlflow ui --port 5001
```
Then visit: http://localhost:5001

### Issue 5: Synthetic data script seems to hang
**Solution:** This is normal! It's creating the data. Wait 1-2 minutes. You'll see "✓ Synthetic data created!" when done.

### Issue 6: Training is very slow
**Solution:** 
- Use smaller batch size: `--batch-size 4`
- Use fewer epochs: `--epochs 2`
- For testing, use smaller data: edit the synthetic data creation to generate only 20 examples instead of 100

---

## 📚 Quick Reference Commands

### Essential Commands (Copy-Paste Ready)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup MLflow
python scripts/setup_mlflow.py

# 3. Create synthetic data
python -c "from scripts.prepare_data import create_synthetic_ner_data, create_synthetic_classification_data; import json; from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True); ner_data = create_synthetic_ner_data(100); clf_data = create_synthetic_classification_data(100); json.dump(ner_data, open('data/raw/ner_data.json', 'w'), indent=2); json.dump(clf_data, open('data/raw/classification_data.json', 'w'), indent=2); print('✓ Synthetic data created!')"

# 4. Prepare data
python -c "from scripts.prepare_data import load_ner_data, load_classification_data, split_data, create_ner_label_vocab, create_classification_label_vocab; import json; from pathlib import Path; ner_data = load_ner_data('data/raw/ner_data.json'); clf_data = load_classification_data('data/raw/classification_data.json'); ner_train, ner_val, ner_test = split_data(ner_data); clf_train, clf_val, clf_test = split_data(clf_data); Path('data/processed').mkdir(parents=True, exist_ok=True); json.dump(ner_train, open('data/processed/ner_train.json', 'w')); json.dump(ner_val, open('data/processed/ner_val.json', 'w')); json.dump(ner_test, open('data/processed/ner_test.json', 'w')); json.dump(clf_train, open('data/processed/clf_train.json', 'w')); json.dump(clf_val, open('data/processed/clf_val.json', 'w')); json.dump(clf_test, open('data/processed/clf_test.json', 'w')); ner_vocab = create_ner_label_vocab(ner_data); clf_vocab = create_classification_label_vocab(clf_data); json.dump(ner_vocab, open('data/processed/ner_labels.json', 'w')); json.dump(clf_vocab, open('data/processed/clf_labels.json', 'w')); print('✓ Data prepared!')"

# 5. Train NER model
python scripts/train_ner.py --data-path data/processed --save-dir models/ner --model-type transformer --model-name bert-base-uncased --batch-size 8 --epochs 3 --lr 2e-5 --use-mlflow --run-name my_first_ner_model

# 6. View results
python scripts/setup_mlflow.py --start-ui
# Then open: http://localhost:5000
```

---

## 🎓 Learning Path

### If you're just starting:
1. Follow Steps 1-8 exactly as written
2. Use synthetic data (Option A)
3. Train with small settings (batch-size=4, epochs=2)
4. Explore MLflow UI to understand the results

### If you're more experienced:
1. Follow Steps 1-4
2. Get real medical data (Option B)
3. Increase training parameters (batch-size=16, epochs=10)
4. Experiment with different models and hyperparameters

---

## 📖 Additional Resources

- **MLOps Guide**: `docs/mlops.md` - Detailed MLOps workflows
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md` - Technical details
- **Quick Start**: `QUICKSTART.md` - 5-minute quick reference

### Get Help
```bash
# View help for any script
python scripts/train_ner.py --help
python scripts/train_classifier.py --help
python scripts/model_registry.py --help
```

---

## ✅ Final Checklist

Before you start, make sure you have:
- [ ] Python 3.8+ installed
- [ ] Project folder open in terminal
- [ ] Internet connection (for downloading models)
- [ ] At least 4GB free disk space

After completing this guide, you should have:
- [ ] All dependencies installed
- [ ] MLflow running
- [ ] Training data created
- [ ] At least one model trained
- [ ] MLflow UI accessible
- [ ] Understanding of how to train more models

---

## 🎉 Congratulations!

You've successfully:
- ✅ Set up the complete environment
- ✅ Created/obtained training data
- ✅ Trained your first medical AI model
- ✅ Tracked experiments with MLflow

**You're now ready to:**
- Train models with different parameters
- Use real medical datasets
- Deploy models via API
- Monitor model performance

**Happy training! 🚀**

---

## 💡 Pro Tips

1. **Start small**: Use small datasets and few epochs for testing
2. **Monitor training**: Always check MLflow UI to see how training progresses
3. **Experiment freely**: Try different learning rates and batch sizes
4. **Save your work**: MLflow automatically saves everything
5. **Read errors carefully**: Error messages usually tell you exactly what's wrong

**Need help?** Check the error message, consult the documentation files, or review the test files for examples.
