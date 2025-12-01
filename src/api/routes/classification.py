"""
Classification API endpoints.
"""

import time
from typing import List, Dict

from fastapi import APIRouter, HTTPException, status

from src.models.classification import ClinicalClassifierSystem
from src.api.models import ClassificationRequest, ClassificationResponse, ClassificationPrediction
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Global model instance (lazy loaded)
_classifier = None


def get_classifier():
    """Get or initialize classifier model."""
    global _classifier
    if _classifier is None:
        logger.info("Loading Clinical Classifier model...")
        try:
            _classifier = ClinicalClassifierSystem()
            logger.info("Clinical Classifier model loaded")
        except Exception as e:
            logger.warning(f"Could not load Clinical Classifier: {e}")
            _classifier = MockClassifier()
    return _classifier


class MockClassifier:
    """Mock classifier for demo purposes."""
    
    def __init__(self):
        logger.info("Using Clinical Classifier in DEMO MODE")
    
    def predict(self, text: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict]:
        """Mock classification using keyword matching."""
        import re
        
        # Simple keyword-based classification matching trained model labels
        categories = {
            # Medical Specialties
            "cardiology": ["chest pain", "heart", "cardiac", "hypertension", "blood pressure", "troponin", "ecg", "catheterization"],
            "neurology": ["headache", "stroke", "seizure", "brain", "neurological", "mri", "ct scan"],
            "oncology": ["cancer", "tumor", "malignancy", "metastasis", "chemotherapy", "radiation"],
            "general_medicine": ["follow-up", "routine", "checkup", "general", "primary care"],
            # Document Types
            "progress_note": ["progress", "follow-up", "visit", "presents with", "assessment", "plan"],
            "discharge_summary": ["discharge", "discharged", "admitted", "hospital course", "disposition"],
            "consultation": ["consultation", "consult", "requested", "opinion", "referred"],
            "admission_note": ["admission", "admitted", "presenting complaint", "chief complaint", "history of present illness"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)
            if score > 0:
                scores[category] = min(score * 2, 0.95)  # Normalize
        
        # Sort by score and return top_k
        results = [
            {"label": cat, "score": score, "probability": score}
            for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score >= threshold
        ][:top_k]
        
        # Default if no matches
        if not results:
            results = [{"label": "general_medicine", "score": 0.5, "probability": 0.5}]
        
        return results


@router.post("/predict", response_model=ClassificationResponse)
async def classify_text(
    request: ClassificationRequest
):
    """
    Classify clinical text into categories.
    
    Args:
        request: Classification request
        
    Returns:
        ClassificationResponse: Classification predictions
    """
    try:
        start_time = time.time()
        
        # Get classifier
        classifier = get_classifier()
        
        logger.info("Classifying text")
        
        # Get predictions with probabilities
        label_probs = classifier.classify(
            request.text,
            return_probabilities=True
        )
        
        # Sort by probability and filter by threshold
        sorted_predictions = sorted(
            label_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply threshold and top_k
        predictions = [
            {"label": label, "score": float(prob), "probability": float(prob)}
            for label, prob in sorted_predictions
            if prob >= request.threshold
        ][:request.top_k]
        
        # Convert to API format
        api_predictions = [
            ClassificationPrediction(
                label=pred["label"],
                score=pred["score"],
                probability=pred["probability"]
            )
            for pred in predictions
        ]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResponse(
            predictions=api_predictions,
            text=request.text,
            processing_time_ms=round(processing_time, 2),
            model_used="Clinical Classifier"
        )
    
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/categories")
async def list_categories():
    """
    List available classification categories.
    
    Returns:
        Dict: Available categories and descriptions
    """
    # Return categories matching the trained model's actual labels
    return {
        "categories": [
            # Document Types
            {
                "id": "progress_note",
                "name": "Progress Note",
                "description": "Clinical progress notes documenting patient visits and ongoing care",
                "type": "document_type"
            },
            {
                "id": "discharge_summary",
                "name": "Discharge Summary",
                "description": "Summary of hospital stay and discharge instructions",
                "type": "document_type"
            },
            {
                "id": "consultation",
                "name": "Consultation",
                "description": "Specialist consultation notes and recommendations",
                "type": "document_type"
            },
            {
                "id": "admission_note",
                "name": "Admission Note",
                "description": "Initial assessment and admission documentation",
                "type": "document_type"
            },
            # Medical Specialties
            {
                "id": "cardiology",
                "name": "Cardiology",
                "description": "Heart and cardiovascular conditions",
                "type": "specialty"
            },
            {
                "id": "neurology",
                "name": "Neurology",
                "description": "Brain and nervous system conditions",
                "type": "specialty"
            },
            {
                "id": "oncology",
                "name": "Oncology",
                "description": "Cancer and tumor-related conditions",
                "type": "specialty"
            },
            {
                "id": "general_medicine",
                "name": "General Medicine",
                "description": "General medical conditions and primary care",
                "type": "specialty"
            }
        ],
        "total_categories": 8
    }


@router.get("/model/info")
async def model_info():
    """
    Get classifier model information.
    
    Returns:
        Dict: Model information and statistics
    """
    return {
        "model_name": "Clinical Classifier",
        "model_type": "transformer_based",
        "base_model": "medicalai/ClinicalBERT",
        "status": "loaded" if _classifier else "not_loaded",
        "capabilities": [
            "Multi-label classification",
            "Confidence scoring",
            "Top-k predictions",
            "Document type classification",
            "Medical specialty classification"
        ],
        "total_labels": 8,
        "label_types": {
            "document_types": 4,
            "specialties": 4
        }
    }
