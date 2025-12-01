# Medical AI Assistant API

## Overview
RESTful API for the Medical AI Assistant system, providing endpoints for Named Entity Recognition (NER), Clinical Classification, RAG-based Question Answering, and Safety Guardrails.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export JWT_SECRET_KEY="your-secret-key-here"
export OPENAI_API_KEY="your-openai-key"  # Optional
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
```

### Running the Server

```bash
# Development mode (with auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Build image
docker build -t medical-ai-api .

# Run container
docker run -p 8000:8000 -e JWT_SECRET_KEY=your-secret medical-ai-api
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Authentication

The API supports two authentication methods:

### 1. JWT Token Authentication

```python
import requests

# Login to get token
response = requests.post(
    "http://localhost:8000/api/v1/auth/token",
    json={"username": "demo", "password": "demo123"}
)
token = response.json()["access_token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/api/v1/ner/extract",
    headers=headers,
    json={"text": "Patient has diabetes"}
)
```

### 2. API Key Authentication

```python
import requests

# Use API key in header
headers = {"X-API-Key": "your-api-key"}
response = requests.post(
    "http://localhost:8000/api/v1/ner/extract",
    headers=headers,
    json={"text": "Patient has diabetes"}
)
```

## API Endpoints

### Health & Monitoring

#### GET /health
Health check endpoint
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "services": {
    "ner": "loaded",
    "classification": "loaded",
    "rag": "loaded",
    "safety": "loaded"
  }
}
```

#### GET /metrics
Prometheus metrics endpoint

#### GET /ready
Readiness probe for Kubernetes

#### GET /live
Liveness probe for Kubernetes

### Named Entity Recognition (NER)

#### POST /api/v1/ner/extract
Extract medical entities from text

**Request:**
```json
{
  "text": "Patient has hypertension and takes lisinopril 10mg daily.",
  "model_type": "transformer",
  "include_linking": false
}
```

**Response:**
```json
{
  "entities": [
    {
      "text": "hypertension",
      "label": "DISEASE",
      "start": 12,
      "end": 24,
      "score": 0.98
    },
    {
      "text": "lisinopril",
      "label": "MEDICATION",
      "start": 36,
      "end": 46,
      "score": 0.95
    },
    {
      "text": "10mg",
      "label": "DOSAGE",
      "start": 47,
      "end": 51,
      "score": 0.92
    }
  ],
  "text": "Patient has hypertension and takes lisinopril 10mg daily.",
  "processing_time_ms": 125.5,
  "model_used": "Transformer NER"
}
```

#### POST /api/v1/ner/extract/batch
Batch entity extraction

#### GET /api/v1/ner/models
List available NER models

### Clinical Classification

#### POST /api/v1/classification/predict
Classify clinical text

**Request:**
```json
{
  "text": "Patient presents with chest pain and elevated troponin.",
  "top_k": 3,
  "threshold": 0.5
}
```

**Response:**
```json
{
  "predictions": [
    {
      "label": "cardiology",
      "score": 0.92,
      "probability": 0.92
    },
    {
      "label": "emergency_medicine",
      "score": 0.78,
      "probability": 0.78
    },
    {
      "label": "internal_medicine",
      "score": 0.65,
      "probability": 0.65
    }
  ],
  "text": "Patient presents with chest pain...",
  "processing_time_ms": 89.2,
  "model_used": "Clinical Classifier"
}
```

#### GET /api/v1/classification/categories
List available classification categories

#### GET /api/v1/classification/model/info
Get model information

### RAG (Question Answering)

#### POST /api/v1/rag/ask
Ask a medical question

**Request:**
```json
{
  "query": "What are the first-line treatments for hypertension?",
  "top_k": 5,
  "use_chain_of_thought": false,
  "conversation_id": null
}
```

**Response:**
```json
{
  "answer": "First-line treatments for hypertension include...",
  "query": "What are the first-line treatments for hypertension?",
  "retrieved_documents": [
    {
      "content": "Hypertension treatment guidelines...",
      "score": 0.89,
      "metadata": {"source": "clinical_guidelines.pdf"}
    }
  ],
  "num_sources": 3,
  "reasoning_steps": null,
  "disclaimers": [
    "This is AI-generated medical information..."
  ],
  "processing_time_ms": 450.8
}
```

#### POST /api/v1/rag/documents/upload
Upload document to RAG system

#### GET /api/v1/rag/system/info
Get RAG system information

### Safety Guardrails

#### POST /api/v1/safety/check
Check text for safety issues

**Request:**
```json
{
  "text": "I'm having severe chest pain",
  "check_type": "query"
}
```

**Response:**
```json
{
  "text": "I'm having severe chest pain",
  "safety_level": "critical",
  "is_safe": false,
  "emergency_detected": true,
  "emergency_level": "CRITICAL",
  "warnings": [
    "Critical emergency detected: Chest pain"
  ],
  "recommendations": [
    "🚨 EMERGENCY: Call 911 immediately",
    "Severe chest pain requires immediate medical attention"
  ],
  "prohibited_claims": [],
  "required_disclaimers": ["emergency"],
  "processing_time_ms": 12.3
}
```

#### GET /api/v1/safety/emergency/criteria
Get emergency detection criteria

#### GET /api/v1/safety/prohibited/claims
Get prohibited medical claims

#### GET /api/v1/safety/disclaimers
Get disclaimer types

#### GET /api/v1/safety/dosage/validation
Get dosage validation information

## Rate Limiting

The API implements token bucket rate limiting:
- Default: 60 requests per minute per client
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`: Maximum requests per minute
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `Retry-After`: Seconds to wait when rate limited

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "Error Type",
  "message": "Human-readable error message",
  "details": "Additional error details"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Client Libraries

### Python Client

```python
from examples.api_client_example import MedicalAIClient

# Initialize client
client = MedicalAIClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Extract entities
result = client.extract_entities("Patient has diabetes")

# Ask question
answer = client.ask_question("What causes hypertension?")

# Check safety
safety = client.check_safety("I'm having chest pain")
```

### cURL Examples

```bash
# NER
curl -X POST http://localhost:8000/api/v1/ner/extract \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has diabetes", "model_type": "transformer"}'

# Classification
curl -X POST http://localhost:8000/api/v1/classification/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Chest pain and dyspnea", "top_k": 3}'

# RAG
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes diabetes?"}'
```

## Configuration

Edit `configs/api_config.yaml` to customize:
- Server settings (host, port, workers)
- Authentication methods
- Rate limiting
- Model settings
- CORS policies
- Logging configuration

## Monitoring

### Prometheus Metrics

Available at `/metrics`:
- `medical_ai_requests_total`: Total requests by endpoint and status
- `medical_ai_request_duration_seconds`: Request duration histogram
- `medical_ai_errors_total`: Total errors by type

### Grafana Dashboard

Import the provided Grafana dashboard for visualization:
- Request rate and latency
- Error rates
- Model performance
- Resource utilization

## Security

### Production Checklist

- [ ] Set strong `JWT_SECRET_KEY` in environment
- [ ] Configure CORS `allow_origins` for specific domains
- [ ] Enable HTTPS/TLS
- [ ] Set up API key rotation policy
- [ ] Configure rate limiting per user/key
- [ ] Enable request logging
- [ ] Set up monitoring alerts
- [ ] Regular security audits
- [ ] Keep dependencies updated

### Default Credentials

**⚠️ CHANGE THESE IN PRODUCTION!**

- Username: `admin` / Password: `admin123`
- Username: `demo` / Password: `demo123`

## Performance

### Optimization Tips

1. **Model Caching**: Models are lazy-loaded and cached
2. **Batch Processing**: Use batch endpoints for multiple texts
3. **Async Processing**: API is fully asynchronous
4. **Connection Pooling**: Reuse client connections
5. **Horizontal Scaling**: Run multiple workers

### Benchmarks

Typical response times (with cached models):
- NER extraction: 50-150ms
- Classification: 30-100ms
- RAG (without LLM): 200-500ms
- RAG (with LLM): 1-3 seconds
- Safety check: 10-30ms

## Troubleshooting

### Models not loading

```bash
# Check logs
tail -f logs/api.log

# Verify dependencies
pip install -r requirements.txt

# Download required models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

### Authentication fails

```bash
# Verify JWT secret is set
echo $JWT_SECRET_KEY

# Check API key format
# Should be: med_ai_<random-string>
```

### High latency

- Check model loading (first request is slower)
- Monitor system resources
- Consider increasing workers
- Enable model caching

## Support

For issues and questions:
- GitHub Issues: [repository-url]
- Documentation: [docs-url]
- Email: support@medical-ai.com

## License

MIT License - See LICENSE file for details
