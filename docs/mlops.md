# MLOps Setup Guide

This guide covers the setup and usage of MLOps tools for the Medical AI Assistant project.

## Overview

The project uses the following MLOps tools:
- **MLflow**: Experiment tracking, model registry, and artifact management
- **DVC**: Data versioning and pipeline management
- **Custom Monitoring**: Performance tracking and drift detection

## Table of Contents

- [Prerequisites](#prerequisites)
- [MLflow Setup](#mlflow-setup)
- [DVC Setup](#dvc-setup)
- [Training Models](#training-models)
- [Model Registry](#model-registry)
- [Model Monitoring](#model-monitoring)
- [Testing](#testing)

## Prerequisites

Install required dependencies:

```bash
pip install mlflow dvc scikit-learn seqeval pytest pytest-cov
```

## MLflow Setup

### 1. Initialize MLflow

Run the setup script to initialize MLflow:

```bash
python scripts/setup_mlflow.py
```

This will:
- Create necessary directories (`mlruns/`, `mlartifacts/`)
- Configure tracking URI
- Create default experiments (ner_training, classification_training)
- Setup model registry
- Generate environment variables

### 2. Start MLflow UI

Start the MLflow UI server:

```bash
python scripts/setup_mlflow.py --start-ui
```

Or run in background:

```bash
python scripts/setup_mlflow.py --start-ui --background
```

Access the UI at: http://localhost:5000

### 3. Environment Variables

Add to your shell profile:

```bash
export MLFLOW_TRACKING_URI=./mlruns
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

Or source the generated file:

```bash
source .env.mlflow
```

## DVC Setup

### 1. Initialize DVC

Run the setup script:

```bash
python scripts/setup_dvc.py
```

This will:
- Initialize DVC in the project
- Configure remote storage
- Setup data tracking
- Create pipeline configuration (dvc.yaml)
- Create params.yaml

### 2. Track Data

Add data directories to DVC:

```bash
dvc add data/raw
dvc add data/processed
dvc add data/medical_literature
```

### 3. Configure Remote Storage

#### Local Storage (Default)
```bash
dvc remote add -d local ./dvc-storage
```

#### AWS S3
```bash
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote access_key_id YOUR_KEY
dvc remote modify s3remote secret_access_key YOUR_SECRET
```

#### Google Cloud Storage
```bash
dvc remote add -d gcs gs://my-bucket/dvc-storage
```

### 4. Commit Changes

```bash
git add .dvc .dvcignore data.dvc dvc.yaml params.yaml
git commit -m "Setup DVC"
```

## Training Models

### 1. Prepare Data

```bash
python scripts/prepare_data.py \
    --raw-data-path data/raw \
    --output-path data/processed \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1
```

### 2. Train NER Model

```bash
python scripts/train_ner.py \
    --data-path data/processed \
    --save-dir models/ner \
    --model-type transformer \
    --model-name bert-base-uncased \
    --batch-size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --use-mlflow \
    --run-name ner_experiment_1
```

### 3. Train Classifier

```bash
python scripts/train_classifier.py \
    --data-path data/processed \
    --save-dir models/classifier \
    --model-name bert-base-uncased \
    --batch-size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --use-mlflow \
    --run-name classifier_experiment_1
```

### 4. Run DVC Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train_ner

# Show pipeline status
dvc status

# View pipeline DAG
dvc dag
```

## Model Registry

### 1. Register Model

Register a trained model:

```bash
python scripts/model_registry.py register \
    --model-path models/ner \
    --name medical_ner \
    --description "NER model for medical entities" \
    --run-id <mlflow-run-id>
```

### 2. Promote Model

Promote model to staging or production:

```bash
# Promote to staging
python scripts/model_registry.py promote \
    --name medical_ner \
    --version 1 \
    --stage Staging

# Promote to production
python scripts/model_registry.py promote \
    --name medical_ner \
    --version 1 \
    --stage Production
```

### 3. Compare Models

Compare two model versions:

```bash
python scripts/model_registry.py compare \
    --name medical_ner \
    --version1 1 \
    --version2 2
```

### 4. Export Model

Export model for deployment:

```bash
python scripts/model_registry.py export \
    --name medical_ner \
    --version 1 \
    --output exported_models/ner_v1
```

### 5. List Models

```bash
python scripts/model_registry.py list
```

## Model Monitoring

### 1. Monitor Performance

```bash
python scripts/monitor_models.py \
    --model-name medical_ner \
    --config configs/mlops_config.yaml \
    --report monitoring_reports/ner_report.json
```

### 2. Monitor in Production

In your inference code:

```python
from scripts.monitor_models import ModelMonitor

# Initialize monitor
monitor = ModelMonitor(
    model_name="medical_ner",
    config_path="configs/mlops_config.yaml"
)

# Setup reference data for drift detection
monitor.drift_detector.fit_reference(
    reference_features,
    feature_names
)

# Monitor batch
report = monitor.monitor_batch(
    predictions=predictions,
    ground_truth=labels,
    features=input_features
)

# Check for alerts
if report['alerts']:
    print(f"Alerts: {report['alerts']}")

# Generate periodic reports
monitor.generate_report(output_path="daily_report.json")
```

### 3. Monitoring Metrics

The monitoring system tracks:
- **Performance Metrics**: Accuracy, F1, precision, recall
- **Data Drift**: Feature distribution changes
- **Model Drift**: Prediction distribution changes
- **PSI**: Population Stability Index

## Testing

### 1. Run All Tests

```bash
pytest tests/
```

### 2. Run Specific Test Files

```bash
# Test NER models
pytest tests/test_ner.py

# Test classification
pytest tests/test_classification.py

# Test RAG
pytest tests/test_rag.py

# Test safety
pytest tests/test_safety.py

# Test API
pytest tests/test_api.py
```

### 3. Run with Coverage

```bash
pytest tests/ --cov=src --cov=scripts --cov-report=html
```

### 4. Run Only Fast Tests

```bash
pytest tests/ -m "not slow"
```

### 5. Run Integration Tests

```bash
pytest tests/ -m integration
```

## MLOps Workflows

### Development Workflow

1. **Develop & Test**
   ```bash
   # Make changes to code
   pytest tests/
   ```

2. **Track Experiments**
   ```bash
   python scripts/train_ner.py --use-mlflow
   ```

3. **Version Data**
   ```bash
   dvc add data/
   git add data.dvc
   git commit -m "Update dataset"
   dvc push
   ```

4. **Register Best Model**
   ```bash
   python scripts/model_registry.py register \
       --model-path models/ner \
       --name medical_ner
   ```

### Deployment Workflow

1. **Promote to Staging**
   ```bash
   python scripts/model_registry.py promote \
       --name medical_ner \
       --version 2 \
       --stage Staging
   ```

2. **Test in Staging**
   ```bash
   # Run integration tests
   pytest tests/test_api.py
   ```

3. **Monitor Performance**
   ```bash
   python scripts/monitor_models.py \
       --model-name medical_ner \
       --report staging_report.json
   ```

4. **Promote to Production**
   ```bash
   python scripts/model_registry.py promote \
       --name medical_ner \
       --version 2 \
       --stage Production
   ```

### Continuous Training Workflow

1. **Detect Drift**
   ```bash
   python scripts/monitor_models.py --model-name medical_ner
   ```

2. **Retrain if Needed**
   ```bash
   dvc repro train_ner
   ```

3. **Compare with Current**
   ```bash
   python scripts/model_registry.py compare \
       --name medical_ner \
       --version1 2 \
       --version2 3
   ```

4. **Deploy if Better**
   ```bash
   python scripts/model_registry.py promote \
       --name medical_ner \
       --version 3 \
       --stage Production
   ```

## Configuration

### MLflow Configuration (configs/mlops_config.yaml)

```yaml
mlflow:
  tracking_uri: ./mlruns
  artifact_location: ./mlartifacts
  backend_store_uri: sqlite:///mlflow.db
  experiments:
    - name: ner_training
      tags:
        model_type: ner
        task: named_entity_recognition
    - name: classification_training
      tags:
        model_type: classifier
        task: multi_label_classification
  server:
    host: 0.0.0.0
    port: 5000
```

### DVC Configuration

```yaml
dvc:
  remotes:
    - name: local
      url: ./dvc-storage
      default: true
  cache:
    dir: .dvc/cache
    type: copy
  data_dirs:
    - data/raw
    - data/processed
    - data/medical_literature
```

### Monitoring Configuration

```yaml
monitoring:
  performance_threshold: 0.1  # 10% degradation
  drift_threshold: 0.1
  psi_threshold: 0.2
  window_days: 7
  alert_email: alerts@example.com
```

## Troubleshooting

### MLflow Issues

**Issue**: Cannot connect to MLflow server
```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Test connection
python scripts/setup_mlflow.py --test-only
```

**Issue**: Experiments not showing
```bash
# List experiments
mlflow experiments list

# Recreate experiments
python scripts/setup_mlflow.py
```

### DVC Issues

**Issue**: DVC not initialized
```bash
dvc init
```

**Issue**: Remote not configured
```bash
dvc remote list
dvc remote add -d local ./dvc-storage
```

**Issue**: Pipeline not running
```bash
# Check status
dvc status

# Debug specific stage
dvc repro --force train_ner
```

## Best Practices

1. **Experiment Tracking**
   - Always use `--use-mlflow` flag when training
   - Use descriptive run names
   - Log hyperparameters and metrics
   - Save model artifacts

2. **Data Versioning**
   - Version data with DVC
   - Commit `.dvc` files to git
   - Push data to remote storage
   - Never commit raw data to git

3. **Model Registry**
   - Register all trained models
   - Use semantic versioning
   - Add descriptions and tags
   - Promote through stages (Staging → Production)

4. **Monitoring**
   - Setup monitoring in production
   - Track performance metrics
   - Detect drift early
   - Generate regular reports

5. **Testing**
   - Write tests for all components
   - Run tests before deployment
   - Use CI/CD for automation
   - Maintain test coverage > 80%

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Model Monitoring Best Practices](https://neptune.ai/blog/ml-model-monitoring-best-practices)

## Support

For issues or questions:
- Check troubleshooting section above
- Review MLflow/DVC documentation
- Open an issue in the project repository
