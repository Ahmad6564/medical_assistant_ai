# 🏥 Medical AI Assistant

A production-ready, end-to-end healthcare AI system demonstrating advanced NLP, ASR, RAG, and MLOps capabilities for clinical decision support.

---

## 🚀 **NEW! Complete Setup Guide for Beginners**

**→ [📖 COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)** ← **START HERE!**

**Perfect for beginners!** This guide includes:
- ✅ Step-by-step instructions in exact order
- ✅ How to get or create training data (no real dataset required!)
- ✅ Simple commands to copy and paste
- ✅ Troubleshooting common issues
- ✅ Explanation of every command

**Quick Start (3 commands):**
```bash
pip install -r requirements.txt
python setup_data.py          # Creates synthetic training data
python scripts/train_ner.py --data-path data/processed --use-mlflow
```

**Other Documentation:**
- [QUICKSTART.md](QUICKSTART.md) - 5-minute quick reference
- [docs/mlops.md](docs/mlops.md) - Complete MLOps guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

---

## 🎯 Project Overview

This comprehensive Medical AI Assistant showcases state-of-the-art AI techniques applied to healthcare, including:

- **Medical Named Entity Recognition (NER)** - Extract diseases, medications, dosages, and anatomical terms
- **Clinical Note Classification** - Multi-label document classification
- **Medical Speech Recognition (ASR)** - Fine-tuned Whisper for medical terminology
- **RAG-based Q&A System** - Retrieval-Augmented Generation with medical literature
- **LLM Integration** - Advanced prompt engineering for clinical reasoning
- **AI Safety Guardrails** - Healthcare-specific safety mechanisms
- **MLOps Pipeline** - Model versioning, tracking, and monitoring
- **Production API** - FastAPI with authentication and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Medical AI Assistant                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   NER    │  │Classifier│  │   ASR    │  │   RAG    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                          │                                    │
│                  ┌───────▼────────┐                          │
│                  │  Safety Layer  │                          │
│                  └───────┬────────┘                          │
│                          │                                    │
│                  ┌───────▼────────┐                          │
│                  │   FastAPI      │                          │
│                  └───────┬────────┘                          │
│                          │                                    │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐           │
│    │ MLflow  │    │Prometheus │   │  Vector   │           │
│    │         │    │           │   │    DB     │           │
│    └─────────┘    └───────────┘   └───────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 1. Medical NER System
- BiLSTM-CRF and Transformer-based models
- Entity types: PROBLEM, TREATMENT, TEST, ANATOMY
- BioClinicalBERT for medical domain
- Entity linking and post-processing

### 2. Clinical Note Classification
- Multi-label classification
- Handles: Progress notes, Discharge summaries, Consultations, Admission notes
- Focal loss for class imbalance
- Per-class performance metrics

### 3. Medical ASR
- Whisper model fine-tuned for medical terminology
- Custom medical vocabulary
- Timestamp extraction
- Medical WER evaluation

### 4. RAG Q&A System
- Hybrid retrieval (dense + sparse)
- Cross-encoder re-ranking
- Query rewriting and expansion
- Source citation support

### 5. AI Safety Guardrails
- Emergency situation detection
- Prohibited medical claims filtering
- Automatic disclaimers
- Dosage validation
- Confidence calibration

### 6. MLOps Pipeline
- Experiment tracking with MLflow
- Model versioning with DVC
- Performance monitoring
- A/B testing infrastructure

### 7. Production API
- FastAPI with OpenAPI documentation
- JWT authentication
- Rate limiting
- Prometheus metrics
- Health checks

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Docker & Docker Compose (optional)
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/ccmuhammadahmad/medical-ai-assistant.git
cd medical-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running the API

**Quick Start (Windows):**
```cmd
start_api.bat
```

**Quick Start (Linux/Mac):**
```bash
chmod +x start_api.sh
./start_api.sh
```

**Manual Start:**
```bash
# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Health Check: http://localhost:8000/health
# - Metrics: http://localhost:8000/metrics
```

See [API_QUICKSTART.md](API_QUICKSTART.md) for detailed setup instructions.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
```

## 📖 Usage Examples

### Medical NER

```python
from src.models.ner import MedicalNER

ner = MedicalNER()
text = "Patient has diabetes mellitus type 2, currently on metformin 500mg BID"
entities = ner.extract_entities(text)

# Output:
# {
#   "PROBLEM": [{"text": "diabetes mellitus type 2", "score": 0.98}],
#   "TREATMENT": [{"text": "metformin", "score": 0.95}],
#   "TEST": []
# }
```

### Clinical Classification

```python
from src.models.classification import ClinicalClassifier

classifier = ClinicalClassifier()
note = "Patient discharged home in stable condition..."
labels = classifier.classify(note)

# Output: ["discharge_summary", "cardiology"]
```

### Medical ASR

```python
from src.models.asr import MedicalASR

asr = MedicalASR()
transcription = asr.transcribe("path/to/audio.wav")

# Output: "Patient presents with chest pain and dyspnea..."
```

### RAG Q&A

```python
from src.rag import MedicalRAG

rag = MedicalRAG()
answer = rag.ask("What are the treatment options for hypertension?")

# Output with sources:
# {
#   "answer": "Treatment options include...",
#   "sources": ["Source 1", "Source 2"],
#   "confidence": 0.92
# }
```

### API Usage

**Python Client:**
```python
from examples.api_client_example import MedicalAIClient

client = MedicalAIClient(base_url="http://localhost:8000", api_key="your-key")
result = client.extract_entities("Patient has hypertension")
print(result['entities'])
```

**cURL Examples:**
```bash
# Extract entities
curl -X POST "http://localhost:8000/api/v1/ner/extract" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has hypertension on lisinopril 10mg daily"}'

# Ask question (RAG)
curl -X POST "http://localhost:8000/api/v1/rag/ask" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes hypertension?"}'

# Safety check
curl -X POST "http://localhost:8000/api/v1/safety/check" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "I have chest pain", "check_type": "query"}'
```

**Full API Documentation:** See [docs/API_GUIDE.md](docs/API_GUIDE.md)

## 🧪 Training Models

### Train NER Model

```bash
python scripts/train_ner.py \ 
  --config configs/ner_config.yaml \ 
  --data data/processed/ner_train.json \ 
  --output models/ner_model
```

### Train Classifier

```bash
python scripts/train_classifier.py \ 
  --config configs/classifier_config.yaml \ 
  --data data/processed/classification_train.json \ 
  --output models/classifier_model
```

### Fine-tune ASR

```bash
python scripts/finetune_asr.py \ 
  --config configs/asr_config.yaml \ 
  --data data/processed/asr_train \ 
  --output models/asr_model
```

## 📊 Evaluation

```bash
# Evaluate all models
python scripts/evaluate_models.py --all

# Evaluate specific model
python scripts/evaluate_models.py --model ner

# Generate evaluation report
python scripts/evaluate_models.py --report --output reports/
```

## 🔧 Configuration

Configuration files are in `configs/` directory:

- `ner_config.yaml` - NER model hyperparameters
- `classifier_config.yaml` - Classification settings
- `asr_config.yaml` - ASR fine-tuning parameters
- `rag_config.yaml` - RAG system configuration
- `api_config.yaml` - API server settings

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_models/test_ner.py -v

# Run integration tests
pytest tests/test_api/ -v
```

## 📈 Monitoring

### MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000

# Access at http://localhost:5000
```

### Prometheus & Grafana

```bash
# Start monitoring stack
docker-compose -f deployment/docker/docker-compose.monitoring.yml up

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

## 🚢 Deployment

### Kubernetes

```bash
# Apply Kubernetes configurations
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -n medical-ai

# Access logs
kubectl logs -f deployment/medical-ai-api -n medical-ai
```

### Cloud Deployment

See detailed deployment guides:
- [AWS Deployment](docs/DEPLOYMENT.md#aws)
- [GCP Deployment](docs/DEPLOYMENT.md#gcp)
- [Azure Deployment](docs/DEPLOYMENT.md#azure)

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Evaluation Metrics](docs/EVALUATION_METRICS.md)
- [Safety Guidelines](docs/SAFETY_GUIDELINES.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- BioClinicalBERT by Emily Alsentzer
- OpenAI Whisper
- Hugging Face Transformers
- LangChain
- FastAPI

## 📧 Contact

**Muhammad Ahmad**
- GitHub: [@ccmuhammadahmad](https://github.com/ccmuhammadahmad)
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

## 🎯 Project Status

This project demonstrates production-ready implementation of:
- ✅ Medical NLP (NER, Classification)
- ✅ Speech Recognition (ASR)
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ LLM Integration & Prompt Engineering
- ✅ AI Safety Guardrails
- ✅ MLOps Pipeline
- ✅ Production API
- ✅ Docker & Kubernetes Deployment

Built as a comprehensive portfolio project for Healthcare AI Engineering roles.

---

**⭐ If you find this project helpful, please consider giving it a star!**