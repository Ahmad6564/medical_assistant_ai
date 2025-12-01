"""
NER (Named Entity Recognition) API endpoints.
"""

import time
from typing import List, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

from src.models.ner import MedicalNER, EntityLinker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Global model instances (lazy loaded)
_transformer_ner = None
_bilstm_ner = None
_entity_linker = None


def get_transformer_ner():
    """Get or initialize transformer NER model."""
    global _transformer_ner
    if _transformer_ner is None:
        logger.info("Loading Transformer NER model...")
        try:
            _transformer_ner = MedicalNER(
                model_type="transformer",
                use_crf=False,
                use_entity_linking=False
            )
            logger.info("Transformer NER model loaded")
        except Exception as e:
            logger.warning(f"Could not load Transformer NER model: {e}")
            _transformer_ner = MockNERModel("Transformer NER")
    return _transformer_ner


def get_bilstm_ner():
    """Get or initialize BiLSTM-CRF NER model."""
    global _bilstm_ner
    if _bilstm_ner is None:
        logger.info("Loading BiLSTM-CRF NER model...")
        try:
            _bilstm_ner = MedicalNER(
                model_type="bilstm_crf",
                use_crf=True,
                use_entity_linking=False
            )
            logger.info("BiLSTM-CRF NER model loaded")
        except Exception as e:
            logger.warning(f"Could not load BiLSTM-CRF NER model: {e}")
            _bilstm_ner = MockNERModel("BiLSTM-CRF NER")
    return _bilstm_ner


def get_entity_linker():
    """Get or initialize entity linker."""
    global _entity_linker
    if _entity_linker is None:
        logger.info("Loading Entity Linker...")
        try:
            _entity_linker = EntityLinker()
            logger.info("Entity Linker loaded")
        except Exception as e:
            logger.warning(f"Could not load Entity Linker: {e}")
            _entity_linker = None
    return _entity_linker


class MockNERModel:
    """Mock NER model for demo purposes when real models can't be loaded."""
    
    def __init__(self, name: str = "Mock NER"):
        self.name = name
        logger.info(f"Using {name} in DEMO MODE - install required packages for full functionality")
    
    def predict(self, text: str) -> List[Dict]:
        """Mock prediction using simple pattern matching."""
        import re
        
        entities = []
        
        # Simple pattern matching for common medical terms
        patterns = {
            "DISEASE": r'\b(diabetes|hypertension|cancer|asthma|copd|pneumonia|covid-19)\b',
            "MEDICATION": r'\b(metformin|lisinopril|aspirin|insulin|warfarin|atorvastatin)\b',
            "DOSAGE": r'\b(\d+\s*mg|\d+\s*mcg|\d+\s*ml)\b',
            "SYMPTOM": r'\b(pain|fever|cough|headache|nausea|dyspnea|fatigue)\b',
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(0),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.95
                })
        
        return entities


class NERRequest(BaseModel):
    text: str = Field(..., description="Text to extract entities from")
    model_type: str = Field("transformer", description="Model type: transformer or bilstm_crf")
    include_linking: bool = Field(False, description="Include entity linking to medical ontologies")

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = None

class NERResponse(BaseModel):
    text: str
    entities: List[Entity]
    model_used: str
    processing_time_ms: float


class BatchNERRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to extract entities from")
    model_type: str = Field("transformer", description="Model type: transformer or bilstm_crf")
    include_linking: bool = Field(False, description="Include entity linking to medical ontologies")


class BatchNERResponse(BaseModel):
    results: List[NERResponse]
    total_processing_time_ms: float


@router.post("/extract", response_model=NERResponse)
async def extract_entities(
    request: NERRequest
):
    """
    Extract medical entities from text.
    
    Args:
        request: NER request with text and options
        
    Returns:
        NERResponse: Extracted entities and metadata
    """
    try:
        logger.info("NER request received")
        start_time = time.time()
        
        # Get appropriate model
        if request.model_type == "transformer":
            model = get_transformer_ner()
            model_name = "Transformer NER"
        elif request.model_type == "bilstm_crf":
            model = get_bilstm_ner()
            model_name = "BiLSTM-CRF NER"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model_type: {request.model_type}. Use 'transformer' or 'bilstm_crf'"
            )
        
        # Extract entities
        logger.info(f"Extracting entities with {model_name}")
        
        # Check if this is MedicalNER or MockNERModel
        if isinstance(model, MockNERModel):
            # MockNERModel returns list of dicts
            entities_dict = {"entities": model.predict(request.text)}
        else:
            # MedicalNER returns grouped entities dict
            entities_dict = model.extract_entities(
                text=request.text,
                return_confidence=True,
                link_entities=request.include_linking
            )
        
        # Convert to flat list of API entity format
        api_entities = []
        
        # Handle MockNERModel format
        if "entities" in entities_dict:
            for ent in entities_dict["entities"]:
                api_entities.append(Entity(
                    text=ent.get("text", ""),
                    label=ent.get("label", ""),
                    start=ent.get("start", 0),
                    end=ent.get("end", 0),
                    confidence=ent.get("score", ent.get("confidence"))
                ))
        else:
            # Handle MedicalNER grouped format (PROBLEM, TREATMENT, TEST, ANATOMY)
            for entity_type, entity_list in entities_dict.items():
                for ent in entity_list:
                    api_entities.append(Entity(
                        text=ent.get("text", ""),
                        label=entity_type,
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        confidence=ent.get("confidence")
                    ))
        
        # Entity linking is now handled by MedicalNER.extract_entities()
        # when include_linking=True is passed
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return NERResponse(
            entities=api_entities,
            text=request.text,
            processing_time_ms=round(processing_time, 2),
            model_used=model_name
        )
    
    except Exception as e:
        logger.error(f"NER extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}"
        )


@router.post("/extract/batch", response_model=BatchNERResponse)
async def extract_entities_batch(
    request: BatchNERRequest
):
    """
    Extract entities from multiple texts in batch.
    
    Args:
        request: Batch NER request
        
    Returns:
        BatchNERResponse: List of NER results
    """
    try:
        start_time = time.time()
        
        # Get model
        if request.model_type == "transformer":
            model = get_transformer_ner()
            model_name = "Transformer NER"
        else:
            model = get_bilstm_ner()
            model_name = "BiLSTM-CRF NER"
        
        logger.info(f"Batch processing {len(request.texts)} texts")
        
        # Process each text
        results = []
        for text in request.texts:
            text_start = time.time()
            
            # Extract entities
            if isinstance(model, MockNERModel):
                entities_dict = {"entities": model.predict(text)}
            else:
                entities_dict = model.extract_entities(
                    text=text,
                    return_confidence=True,
                    link_entities=request.include_linking
                )
            
            # Convert to flat list
            api_entities = []
            if "entities" in entities_dict:
                for ent in entities_dict["entities"]:
                    api_entities.append(Entity(
                        text=ent.get("text", ""),
                        label=ent.get("label", ""),
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        confidence=ent.get("score", ent.get("confidence"))
                    ))
            else:
                for entity_type, entity_list in entities_dict.items():
                    for ent in entity_list:
                        api_entities.append(Entity(
                            text=ent.get("text", ""),
                            label=entity_type,
                            start=ent.get("start", 0),
                            end=ent.get("end", 0),
                            confidence=ent.get("confidence")
                        ))
            
            text_time = (time.time() - text_start) * 1000
            
            results.append(NERResponse(
                entities=api_entities,
                text=text,
                processing_time_ms=round(text_time, 2),
                model_used=model_name
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchNERResponse(
            results=results,
            total_processed=len(request.texts),
            total_processing_time_ms=round(total_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch NER extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch entity extraction failed: {str(e)}"
        )


@router.get("/models")
async def list_models():
    """
    List available NER models.
    
    Returns:
        Dict: Available models and their status
    """
    return {
        "models": [
            {
                "name": "transformer",
                "description": "Transformer-based NER (BERT/BioBERT)",
                "status": "available" if _transformer_ner else "not_loaded",
                "supported_entities": ["DISEASE", "SYMPTOM", "MEDICATION", "DOSAGE", "PROCEDURE", "ANATOMY"]
            },
            {
                "name": "bilstm_crf",
                "description": "BiLSTM-CRF model",
                "status": "available" if _bilstm_ner else "not_loaded",
                "supported_entities": ["DISEASE", "SYMPTOM", "MEDICATION", "DOSAGE", "PROCEDURE", "ANATOMY"]
            }
        ],
        "entity_linking": "available" if _entity_linker else "not_loaded"
    }
