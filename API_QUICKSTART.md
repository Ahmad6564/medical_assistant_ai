# Production API - Quick Start Guide

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip
- Git

### Option 1: Local Development

#### Windows
```cmd
# Clone or navigate to project
cd medical-ai-assistant

# Run startup script (creates venv, installs dependencies, starts server)
start_api.bat
```

#### Linux/Mac
```bash
# Clone or navigate to project
cd medical-ai-assistant

# Make script executable
chmod +x start_api.sh

# Run startup script
./start_api.sh
```

#### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export JWT_SECRET_KEY="your-secret-key"
export OPENAI_API_KEY="your-openai-key"  # Optional

# Start server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
# Build and run with Docker Compose (includes Redis, PostgreSQL, Prometheus, Grafana)
docker-compose up -d

# Or build standalone container
docker build -t medical-ai-api .
docker run -p 8000:8000 -e JWT_SECRET_KEY=your-secret medical-ai-api
```

### Option 3: Production Deployment

```bash
# Install production ASGI server
pip install gunicorn

# Run with Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## 🔗 Access Points

Once running, access:

- **API Documentation (Swagger)**: http://localhost:8000/docs
- **API Documentation (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics (Prometheus)**: http://localhost:8000/metrics
- **Grafana Dashboard**: http://localhost:3000 (Docker only, admin/admin)

## 🔐 Authentication

### Default Credentials (⚠️ CHANGE IN PRODUCTION!)

**JWT Authentication:**
- Admin: `admin` / `admin123`
- Demo: `demo` / `demo123`

**Login Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}'
```

### API Key Authentication

Create API keys programmatically:
```python
from src.api.auth import APIKeyManager

api_key = APIKeyManager.create_api_key(
    name="my-app",
    permissions=["ner", "classification", "rag"]
)
print(f"API Key: {api_key}")
```

Use API key in requests:
```bash
curl -X POST http://localhost:8000/api/v1/ner/extract \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text":"Patient has diabetes"}'
```

## 📋 API Endpoints Overview

### Health & Monitoring
- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `GET /live` - Liveness probe
- `GET /metrics` - Prometheus metrics

### NER (Named Entity Recognition)
- `POST /api/v1/ner/extract` - Extract entities
- `POST /api/v1/ner/extract/batch` - Batch extraction
- `GET /api/v1/ner/models` - List models

### Classification
- `POST /api/v1/classification/predict` - Classify text
- `GET /api/v1/classification/categories` - List categories
- `GET /api/v1/classification/model/info` - Model info

### RAG (Question Answering)
- `POST /api/v1/rag/ask` - Ask question
- `POST /api/v1/rag/documents/upload` - Upload document
- `GET /api/v1/rag/system/info` - System info

### Safety Guardrails
- `POST /api/v1/safety/check` - Safety check
- `GET /api/v1/safety/emergency/criteria` - Emergency criteria
- `GET /api/v1/safety/prohibited/claims` - Prohibited claims
- `GET /api/v1/safety/disclaimers` - Disclaimer types

## 💡 Quick Examples

### 1. Extract Medical Entities (NER)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/ner/extract",
    headers={"X-API-Key": "your-api-key"},
    json={
        "text": "Patient diagnosed with hypertension, prescribed lisinopril 10mg daily",
        "model_type": "transformer"
    }
)

print(response.json())
# Output: {"entities": [{"text": "hypertension", "label": "DISEASE", ...}, ...]}
```

### 2. Classify Clinical Text

```python
response = requests.post(
    "http://localhost:8000/api/v1/classification/predict",
    headers={"X-API-Key": "your-api-key"},
    json={
        "text": "Patient presents with chest pain and elevated troponin levels",
        "top_k": 3
    }
)

print(response.json())
# Output: {"predictions": [{"label": "cardiology", "probability": 0.92}, ...]}
```

### 3. Ask Medical Question (RAG)

```python
response = requests.post(
    "http://localhost:8000/api/v1/rag/ask",
    headers={"X-API-Key": "your-api-key"},
    json={
        "query": "What are the first-line treatments for hypertension?",
        "top_k": 5
    }
)

print(response.json())
# Output: {"answer": "First-line treatments include...", "num_sources": 3, ...}
```

### 4. Check Safety

```python
response = requests.post(
    "http://localhost:8000/api/v1/safety/check",
    headers={"X-API-Key": "your-api-key"},
    json={
        "text": "I'm having severe chest pain and difficulty breathing",
        "check_type": "query"
    }
)

print(response.json())
# Output: {"safety_level": "critical", "emergency_detected": true, ...}
```

### 5. Using Python Client

```python
from examples.api_client_example import MedicalAIClient

# Initialize client
client = MedicalAIClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Extract entities
result = client.extract_entities("Patient has type 2 diabetes")
print(result['entities'])

# Ask question
answer = client.ask_question("What causes hypertension?")
print(answer['answer'])
```

## 🔧 Configuration

Edit `configs/api_config.yaml`:

```yaml
# Server settings
host: "0.0.0.0"
port: 8000
workers: 4

# Rate limiting
rate_limit:
  requests_per_minute: 60

# Model settings
models:
  ner:
    default_model: "transformer"
  rag:
    use_hybrid_search: true
    use_reranker: true
```

## 📊 Monitoring

### Prometheus Metrics

Available at `/metrics`:
- Request count and latency
- Error rates
- Model performance

### Grafana Dashboards

Access at http://localhost:3000 (Docker setup):
- Real-time request monitoring
- Error rate tracking
- Performance metrics

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Windows - Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Dependencies Not Installing
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies with verbose output
pip install -r requirements.txt -v
```

### Models Not Loading
```bash
# Download required models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
python -m spacy download en_core_web_sm
```

### Authentication Errors
```bash
# Set JWT secret
export JWT_SECRET_KEY="your-secure-secret-key"

# Verify API key format (should start with med_ai_)
echo $X_API_KEY
```

## 🚀 Performance Tips

1. **Use Batch Endpoints**: Process multiple items in one request
2. **Enable Caching**: Models are cached after first load
3. **Horizontal Scaling**: Run multiple workers (`--workers 4`)
4. **Use Redis**: Enable caching with Redis (Docker setup includes it)
5. **Optimize Top-K**: Lower top_k values for faster RAG responses

## 📖 Full Documentation

- **Complete API Guide**: `docs/API_GUIDE.md`
- **API Client Examples**: `examples/api_client_example.py`
- **Configuration Reference**: `configs/api_config.yaml`
- **Docker Setup**: `docker-compose.yml`

## 🛡️ Security Checklist

Before production deployment:

- [ ] Change default passwords and JWT secret
- [ ] Configure CORS for specific domains
- [ ] Enable HTTPS/TLS
- [ ] Set up API key rotation
- [ ] Configure rate limiting per user
- [ ] Enable request logging
- [ ] Set up monitoring alerts
- [ ] Regular security audits

## 📝 Next Steps

1. **Start the server**: Run `start_api.bat` or `start_api.sh`
2. **Test endpoints**: Visit http://localhost:8000/docs
3. **Try examples**: Run `python examples/api_client_example.py`
4. **Read full docs**: See `docs/API_GUIDE.md`
5. **Deploy to production**: Use Docker or Kubernetes

## 💬 Support

- Issues: Open a GitHub issue
- Documentation: Check `docs/` directory
- Examples: See `examples/` directory

---

**Ready to go!** Run the startup script and start building with the Medical AI Assistant API! 🏥🤖
