"""
Safety guardrails API endpoints.
"""

import time

from fastapi import APIRouter, HTTPException, status

from src.safety import MedicalAISafetyGuardrails
from src.api.models import SafetyCheckRequest, SafetyCheckResponse, SafetyLevel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Global instance (lazy loaded)
_safety_guardrails = None


def get_safety_guardrails():
    """Get or initialize safety guardrails."""
    global _safety_guardrails
    if _safety_guardrails is None:
        logger.info("Loading safety guardrails...")
        try:
            _safety_guardrails = MedicalAISafetyGuardrails(
                enable_emergency_detection=True,
                enable_claim_filtering=True,
                enable_dosage_validation=True,
                enable_disclaimers=True,
                strict_mode=True
            )
            logger.info("Safety guardrails loaded")
        except Exception as e:
            logger.warning(f"Could not load safety guardrails: {e}. Using mock mode.")
            _safety_guardrails = MockSafetyGuardrails()
    return _safety_guardrails


class MockSafetyGuardrails:
    """Mock safety system for demo purposes."""
    
    def __init__(self):
        logger.info("Using Safety Guardrails in DEMO MODE")
    
    def check_query_safety(self, text: str):
        """Mock safety check."""
        from src.safety.guardrails import SafetyCheckResult, SafetyLevel, DisclaimerType
        
        # Simple keyword-based emergency detection
        emergency_keywords = ["chest pain", "can't breathe", "suicide", "overdose", "severe bleeding"]
        is_emergency = any(kw in text.lower() for kw in emergency_keywords)
        
        return SafetyCheckResult(
            is_safe=not is_emergency,
            safety_level=SafetyLevel.CRITICAL if is_emergency else SafetyLevel.SAFE,
            emergency_detected=is_emergency,
            warnings=["Emergency detected"] if is_emergency else [],
            recommendations=["Call 911 immediately"] if is_emergency else [],
            prohibited_claims=[],
            dosage_issues=[],
            required_disclaimers=[DisclaimerType.EMERGENCY] if is_emergency else [],
            filtered_content=None,
            metadata={}
        )
    
    def check_response_safety(self, text: str):
        """Mock response safety check."""
        return self.check_query_safety(text)
    
    def safe_response(self, query: str, response: str):
        """Mock safe response."""
        query_result = self.check_query_safety(query)
        response_result = self.check_response_safety(response)
        
        is_safe = query_result.is_safe and response_result.is_safe
        
        if not is_safe:
            response = "⚠️ EMERGENCY DETECTED - Please call 911 or seek immediate medical attention."
        
        return is_safe, response, response_result


@router.post("/check", response_model=SafetyCheckResponse)
async def check_safety(
    request: SafetyCheckRequest
):
    """
    Check text for safety issues.
    
    Args:
        request: Safety check request
        
    Returns:
        SafetyCheckResponse: Safety analysis results
    """
    try:
        start_time = time.time()
        
        # Get safety system
        safety = get_safety_guardrails()
        
        logger.info(f"Safety check requested")
        
        # Perform safety checks based on type
        if request.check_type == "query":
            safety_result = safety.check_query_safety(request.text)
        elif request.check_type == "response":
            safety_result = safety.check_response_safety(request.text)
        else:  # both
            # Check as both query and response
            query_result = safety.check_query_safety(request.text)
            response_result = safety.check_response_safety(request.text)
            
            # Combine results (use most severe)
            safety_result = query_result
            if response_result.safety_level.value > query_result.safety_level.value:
                safety_result = response_result
        
        # Convert to API model
        processing_time = (time.time() - start_time) * 1000
        
        # Determine emergency level based on safety level and emergency detection
        emergency_level = None
        if safety_result.emergency_detected:
            if safety_result.safety_level == SafetyLevel.CRITICAL:
                emergency_level = "critical"
            elif safety_result.safety_level == SafetyLevel.HIGH:
                emergency_level = "urgent"
            else:
                emergency_level = "moderate"
        
        return SafetyCheckResponse(
            text=request.text,
            safety_level=SafetyLevel(safety_result.safety_level.value),
            is_safe=safety_result.is_safe,
            emergency_detected=safety_result.emergency_detected,
            emergency_level=emergency_level,
            warnings=safety_result.warnings,
            recommendations=safety_result.recommendations,
            prohibited_claims=safety_result.prohibited_claims,
            required_disclaimers=[d.value for d in safety_result.required_disclaimers],
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Safety check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Safety check failed: {str(e)}"
        )


@router.get("/emergency/criteria")
async def emergency_criteria():
    """
    Get emergency detection criteria.
    
    Returns:
        Dict: Emergency detection criteria
    """
    return {
        "critical_symptoms": [
            "Chest pain",
            "Difficulty breathing",
            "Severe bleeding",
            "Loss of consciousness",
            "Stroke symptoms",
            "Severe allergic reaction",
            "Poisoning/overdose",
            "Suicidal thoughts",
            "Severe pain"
        ],
        "urgent_symptoms": [
            "High fever",
            "Severe headache",
            "Abdominal pain",
            "Back pain",
            "Vomiting",
            "Head injury",
            "Seizure",
            "Severe burns",
            "Broken bones",
            "Eye injury",
            "Urinary problems"
        ],
        "emergency_numbers": {
            "us": "911",
            "uk": "999",
            "eu": "112"
        },
        "crisis_hotlines": {
            "suicide_prevention": "988",
            "crisis_text_line": "Text HOME to 741741"
        }
    }


@router.get("/prohibited/claims")
async def prohibited_claims():
    """
    Get list of prohibited medical claims.
    
    Returns:
        Dict: Prohibited claim categories
    """
    return {
        "categories": [
            {
                "type": "cure_claims",
                "description": "Claims to cure serious diseases",
                "examples": [
                    "This will cure your cancer",
                    "Guaranteed cure for diabetes",
                    "Eliminate HIV completely"
                ]
            },
            {
                "type": "diagnostic_claims",
                "description": "Claims to diagnose conditions",
                "examples": [
                    "You definitely have...",
                    "This is certainly...",
                    "I can diagnose you with..."
                ]
            },
            {
                "type": "dangerous_advice",
                "description": "Potentially harmful recommendations",
                "examples": [
                    "Stop taking your medication",
                    "Avoid seeing a doctor",
                    "Try this instead of treatment"
                ]
            },
            {
                "type": "unproven_treatments",
                "description": "Scientifically unproven therapies",
                "examples": [
                    "Essential oils cure serious diseases",
                    "Miracle supplements",
                    "Alternative treatments without evidence"
                ]
            }
        ],
        "note": "These claims violate medical ethics and regulatory guidelines"
    }


@router.get("/disclaimers")
async def disclaimer_types():
    """
    Get available disclaimer types.
    
    Returns:
        Dict: Disclaimer types and their usage
    """
    return {
        "disclaimer_types": [
            {
                "type": "general",
                "when_used": "All medical information responses",
                "purpose": "Clarify AI limitations"
            },
            {
                "type": "emergency",
                "when_used": "Emergency situations detected",
                "purpose": "Direct to immediate medical care"
            },
            {
                "type": "diagnosis",
                "when_used": "Diagnostic discussions",
                "purpose": "Emphasize need for professional diagnosis"
            },
            {
                "type": "treatment",
                "when_used": "Treatment recommendations",
                "purpose": "Clarify need for medical supervision"
            },
            {
                "type": "medication",
                "when_used": "Medication information",
                "purpose": "Warn about prescription requirements"
            },
            {
                "type": "symptom_check",
                "when_used": "Symptom evaluation",
                "purpose": "Cannot replace medical examination"
            },
            {
                "type": "pediatric",
                "when_used": "Children's health topics",
                "purpose": "Emphasize need for pediatric care"
            },
            {
                "type": "mental_health",
                "when_used": "Mental health discussions",
                "purpose": "Crisis resources and professional care"
            }
        ]
    }


@router.get("/dosage/validation")
async def dosage_validation_info():
    """
    Get dosage validation information.
    
    Returns:
        Dict: Dosage validation capabilities
    """
    return {
        "validated_medications": [
            "Metformin",
            "Lisinopril",
            "Atorvastatin",
            "Levothyroxine",
            "Amlodipine",
            "Omeprazole",
            "Gabapentin",
            "Sertraline",
            "Losartan",
            "Aspirin",
            "Warfarin",
            "Insulin"
        ],
        "validation_checks": [
            "Dosage range verification",
            "Pediatric/geriatric considerations",
            "Narrow therapeutic index warnings",
            "Common drug interactions",
            "Frequency validation"
        ],
        "risk_levels": ["SAFE", "CAUTION", "WARNING", "DANGEROUS"],
        "note": "This is basic validation only. Always consult a healthcare provider."
    }
