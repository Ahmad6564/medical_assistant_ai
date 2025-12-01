# Quick Start Guide - Medical AI Assistant

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 min)

```bash
pip install -r requirements.txt
```

Required packages:
- torch, transformers (ML models)
- mlflow (experiment tracking)
- dvc (data versioning)
- pytest (testing)
- fastapi (API)

### Step 2: Setup MLOps (1 min)

```bash
# Initialize MLflow
python scripts/setup_mlflow.py

# Start MLflow UI (optional)
python scripts/setup_mlflow.py --start-ui --background
```

Access MLflow UI at: http://localhost:5000

### Step 3: Run Tests (1 min)

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_ner.py -v
```

### Step 4: Train a Model (1 min for setup, actual training takes longer)

```bash
# Generate synthetic data for testing
python scripts/prepare_data.py --create-synthetic

# Train NER model with MLflow tracking
python scripts/train_ner.py \
    --data-path data/processed \
    --save-dir models/ner \
    --model-name bert-base-uncased \
    --batch-size 8 \
    --epochs 2 \
    --use-mlflow \
    --run-name quick_test
```

### Step 5: View Results (30 sec)

Open MLflow UI (http://localhost:5000) to see:
- Experiment metrics (F1, precision, recall)
- Training curves
- Model artifacts
- Hyperparameters

---

## 📚 Common Tasks

### Train NER Model
```bash
python scripts/train_ner.py \
    --data-path data/processed \
    --save-dir models/ner \
    --use-mlflow \
    --run-name my_experiment
```

### Train Classifier
```bash
python scripts/train_classifier.py \
    --data-path data/processed \
    --save-dir models/classifier \
    --use-mlflow \
    --run-name my_experiment
```

### Register Model
```bash
python scripts/model_registry.py register \
    --model-path models/ner \
    --name medical_ner \
    --description "My NER model"
```

### Promote to Production
```bash
python scripts/model_registry.py promote \
    --name medical_ner \
    --version 1 \
    --stage Production
```

### Monitor Model
```bash
python scripts/monitor_models.py \
    --model-name medical_ner \
    --report report.json
```

### Run Tests
```bash
# All tests
pytest tests/

# Fast tests only (skip slow ones)
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=src --cov=scripts --cov-report=html
```

---

## 📁 Project Structure

```
medical-ai-assistant/
├── scripts/              # Training & MLOps scripts
│   ├── train_ner.py     # Train NER models
│   ├── train_classifier.py  # Train classifiers
│   ├── setup_mlflow.py  # Setup MLflow
│   ├── model_registry.py    # Manage models
│   └── monitor_models.py    # Monitor performance
│
├── tests/               # Test suite
│   ├── test_ner.py     # NER tests
│   ├── test_classification.py
│   ├── test_rag.py
│   ├── test_safety.py
│   └── test_api.py
│
├── src/                 # Source code
│   ├── models/         # Model implementations
│   ├── api/            # API endpoints
│   ├── rag/            # RAG system
│   └── utils/          # Utilities
│
├── configs/            # Configuration files
├── data/               # Data directory
└── docs/               # Documentation
```

---

## 🔧 Configuration

Edit `configs/mlops_config.yaml`:

```yaml
mlflow:
  tracking_uri: ./mlruns
  server:
    host: 0.0.0.0
    port: 5000

dvc:
  remotes:
    - name: local
      url: ./dvc-storage
      default: true

monitoring:
  performance_threshold: 0.1
  drift_threshold: 0.1
  window_days: 7
```

---

## 🧪 Testing

### Run Specific Tests
```bash
pytest tests/test_ner.py::TestTransformerNER::test_forward_pass -v
```

### Run with Markers
```bash
pytest tests/ -m unit          # Only unit tests
pytest tests/ -m integration   # Only integration tests
pytest tests/ -m "not slow"    # Skip slow tests
```

### Generate Coverage Report
```bash
pytest tests/ --cov=src --cov=scripts --cov-report=html
# Open htmlcov/index.html
```

---

## 📊 MLflow UI

Access at: http://localhost:5000

Features:
- **Experiments**: View all training runs
- **Compare**: Compare different runs
- **Models**: Browse registered models
- **Artifacts**: Download model files

### Common MLflow Commands
```bash
# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-name ner_training

# Serve model
mlflow models serve -m models:/medical_ner/Production -p 5001
```

---

## 🎯 Training Tips

### Best Practices
1. **Always use MLflow**: Add `--use-mlflow` flag
2. **Name your runs**: Use `--run-name descriptive_name`
3. **Start small**: Test with small batch size and few epochs
4. **Monitor training**: Check MLflow UI for metrics

### Hyperparameter Tuning
```bash
# Try different learning rates
for lr in 1e-5 2e-5 3e-5; do
    python scripts/train_ner.py \
        --lr $lr \
        --use-mlflow \
        --run-name "lr_${lr}"
done
```

### GPU Training
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Train with GPU
python scripts/train_ner.py \
    --batch-size 16 \
    --use-mlflow
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'mlflow'"
```bash
pip install mlflow
```

### Issue: "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train_ner.py --batch-size 8
```

### Issue: "MLflow server not starting"
```bash
# Check port availability
python scripts/setup_mlflow.py --test-only

# Use different port
mlflow ui --port 5001
```

### Issue: "Tests failing"
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run with verbose output
pytest tests/ -v -s
```

---

## 📖 Documentation

- **Full MLOps Guide**: `docs/mlops.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Script Help**: Run any script with `--help`
  ```bash
  python scripts/train_ner.py --help
  ```

---

## 🆘 Getting Help

1. **Check documentation**: See `docs/` folder
2. **View logs**: Check console output and MLflow UI
3. **Run tests**: `pytest tests/ -v`
4. **Read error messages**: They usually indicate the problem

---

## ✅ Checklist

Before deploying to production:

- [ ] All tests pass (`pytest tests/`)
- [ ] Models trained with good metrics
- [ ] Models registered in MLflow
- [ ] Production model promoted
- [ ] Monitoring configured
- [ ] API tested (`pytest tests/test_api.py`)
- [ ] Documentation reviewed

---

## 🎉 You're Ready!

You now have:
- ✅ Complete training pipeline
- ✅ Comprehensive test suite
- ✅ MLOps infrastructure
- ✅ Model monitoring
- ✅ Production-ready code

**Start training your models and tracking experiments!**

For detailed information, see:
- `docs/mlops.md` - Complete MLOps guide
- `IMPLEMENTATION_SUMMARY.md` - Full implementation details
