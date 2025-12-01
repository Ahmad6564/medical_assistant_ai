# API Demo Mode

The API is currently running in **DEMO MODE** with mock implementations. This allows you to test the API structure without requiring all the heavy ML models to be downloaded.

## What's Working

✅ **API Server**: FastAPI server is running
✅ **Authentication**: JWT and API key auth working
✅ **All Endpoints**: All routes are accessible
✅ **Documentation**: Swagger UI and ReDoc available
✅ **Mock Responses**: Simple pattern-matching for demos

## Mock Implementations

The following components use mock implementations:

1. **NER (Named Entity Recognition)**
   - Uses regex pattern matching for common medical terms
   - Detects: diseases, medications, dosages, symptoms

2. **Classification**
   - Uses keyword-based category matching
   - Supports: cardiology, neurology, pulmonology, etc.

3. **RAG (Question Answering)**
   - Returns demo responses
   - Document upload tracked but not processed

4. **Safety Guardrails**
   - Basic emergency keyword detection
   - Simple safety checks

## Upgrading to Full Functionality

To enable full ML-powered functionality:

### 1. Install Additional Packages

```bash
pip install torchcrf
```

### 2. Download Pre-trained Models

```bash
# Download BioClinicalBERT
python -c "from transformers import AutoModel; AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Download sentence transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

### 3. Set API Keys (Optional)

For LLM integration (RAG with GPT-4/Claude):

```bash
# Windows
set OPENAI_API_KEY=your-key-here
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
```

### 4. Restart the Server

```bash
# Windows
start_api.bat

# Linux/Mac
./start_api.sh
```

## Current Demo Capabilities

Even in demo mode, you can:

- ✅ Test all API endpoints
- ✅ Verify authentication and permissions
- ✅ Check request/response formats
- ✅ Review API documentation
- ✅ Monitor metrics and health checks
- ✅ Test rate limiting
- ✅ See basic entity extraction
- ✅ Get classification predictions
- ✅ Ask questions (mock responses)
- ✅ Run safety checks

## Testing Demo Mode

```bash
# Health check
curl http://localhost:8000/health

# NER extraction (demo)
curl -X POST http://localhost:8000/api/v1/ner/extract \
  -H "X-API-Key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has diabetes and takes metformin 500mg"}'

# Classification (demo)
curl -X POST http://localhost:8000/api/v1/classification/predict \
  -H "X-API-Key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with chest pain"}'
```

## Production Deployment

For production, complete all upgrade steps above and:

1. Set strong JWT_SECRET_KEY
2. Configure CORS appropriately
3. Enable HTTPS
4. Set up proper model caching
5. Configure database connections
6. Set up monitoring alerts

---

**Note**: Demo mode is perfect for:
- Initial setup and testing
- API structure validation
- Frontend development
- Documentation review
- Performance baseline testing

For full ML capabilities, follow the upgrade steps above! 🚀
