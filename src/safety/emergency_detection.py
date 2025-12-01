"""
Emergency detection system for identifying urgent medical situations.
Detects conditions requiring immediate medical attention.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)


class EmergencyLevel(Enum):
    """Severity levels for emergencies."""
    CRITICAL = "critical"  # Life-threatening, call 911
    URGENT = "urgent"  # Seek immediate medical attention
    NON_URGENT = "non_urgent"  # Can wait for regular appointment
    NONE = "none"  # No emergency detected


@dataclass
class EmergencyDetectionResult:
    """Result from emergency detection."""
    level: EmergencyLevel
    detected_conditions: List[str]
    confidence: float
    recommended_action: str
    reasoning: str
    
    def __bool__(self) -> bool:
        """Allow result to be used in boolean context."""
        return self.level != EmergencyLevel.NONE


class EmergencyDetector:
    """
    Detect medical emergencies from text.
    Identifies symptoms and conditions requiring immediate attention.
    """
    
    # Critical symptoms requiring immediate 911 call
    CRITICAL_SYMPTOMS = {
        # Cardiac
        "chest pain": ["chest pain", "severe chest pain", "crushing chest pain", "chest pressure", "squeezing chest"],
        "heart attack": ["heart attack", "myocardial infarction", "mi"],
        "cardiac arrest": ["cardiac arrest", "heart stopped", "no pulse"],
        
        # Neurological
        "stroke": ["stroke", "cva", "facial drooping", "arm weakness", "speech difficulty"],
        "seizure": ["seizure", "convulsion", "fits", "status epilepticus"],
        "altered consciousness": ["unresponsive", "unconscious", "not waking up", "loss of consciousness"],
        
        # Respiratory
        "breathing difficulty": ["cant breathe", "can't breathe", "cannot breathe", "difficulty breathing", "severe dyspnea", "gasping for air", "breathe properly"],
        "choking": ["choking", "airway obstruction", "can't speak", "cant speak"],
        
        # Trauma
        "severe bleeding": ["severe bleeding", "heavy bleeding", "uncontrolled bleeding", "hemorrhage", "bleeding"],
        "head trauma": ["severe head injury", "head trauma", "skull fracture"],
        "spinal injury": ["spinal injury", "back injury", "paralysis", "cannot move legs"],
        
        # Other critical
        "anaphylaxis": ["anaphylaxis", "severe allergic reaction", "throat swelling", "severe swelling"],
        "overdose": ["overdose", "drug overdose", "poisoning"],
        "suicidal": ["suicidal", "want to die", "suicide attempt", "self-harm", "suicidal thoughts"],
        "severe pain": ["worst pain of my life", "excruciating pain", "10/10 pain"]
    }
    
    # Urgent symptoms requiring immediate medical attention
    URGENT_SYMPTOMS = {
        "chest discomfort": ["chest discomfort", "chest tightness", "angina"],
        "confusion": ["confused", "disoriented", "mental status change"],
        "severe headache": ["worst headache", "sudden severe headache", "thunderclap headache"],
        "visual changes": ["sudden vision loss", "double vision", "blurred vision sudden"],
        "weakness": ["sudden weakness", "arm weakness", "leg weakness", "facial weakness"],
        "severe abdominal pain": ["severe abdominal pain", "acute abdomen"],
        "high fever": ["fever over 103", "temperature above 39", "high fever"],
        "severe vomiting": ["cannot keep fluids down", "severe vomiting", "persistent vomiting"],
        "blood in": ["coughing up blood", "vomiting blood", "blood in stool", "blood in urine"],
        "pregnancy emergency": ["pregnancy bleeding", "severe pregnancy pain"],
        "infant emergency": ["infant not responding", "baby not breathing normally", "infant fever"]
    }
    
    # Emergency keywords
    EMERGENCY_KEYWORDS = [
        "911", "emergency", "ambulance", "critical", "life-threatening",
        "urgent", "immediate", "severe", "acute", "sudden onset"
    ]
    
    def __init__(self, sensitivity: float = 0.7):
        """
        Initialize emergency detector.
        
        Args:
            sensitivity: Detection sensitivity (0-1), higher = more sensitive
        """
        self.sensitivity = sensitivity
        logger.info(f"Initialized EmergencyDetector (sensitivity={sensitivity})")
    
    def detect(self, text: str) -> EmergencyDetectionResult:
        """
        Detect potential emergencies in text.
        
        Args:
            text: Input text (symptoms, description, query)
            
        Returns:
            EmergencyDetectionResult
        """
        text_lower = text.lower()
        
        # Check for critical symptoms
        critical_matches = self._check_symptoms(text_lower, self.CRITICAL_SYMPTOMS)
        
        # Check for urgent symptoms
        urgent_matches = self._check_symptoms(text_lower, self.URGENT_SYMPTOMS)
        
        # Check for emergency keywords
        has_emergency_keywords = any(keyword in text_lower for keyword in self.EMERGENCY_KEYWORDS)
        
        # Determine emergency level
        if critical_matches:
            level = EmergencyLevel.CRITICAL
            detected = list(critical_matches.keys())
            confidence = 0.9
            action = "Call 911 or go to emergency room immediately"
            reasoning = f"Detected critical symptoms: {', '.join(detected)}"
        
        elif urgent_matches and (has_emergency_keywords or len(urgent_matches) > 1):
            level = EmergencyLevel.URGENT
            detected = list(urgent_matches.keys())
            confidence = 0.8
            action = "Seek immediate medical attention at emergency department"
            reasoning = f"Detected urgent symptoms: {', '.join(detected)}"
        
        elif urgent_matches:
            level = EmergencyLevel.NON_URGENT
            detected = list(urgent_matches.keys())
            confidence = 0.6
            action = "Contact healthcare provider promptly or visit urgent care"
            reasoning = f"Detected concerning symptoms: {', '.join(detected)}"
        
        else:
            level = EmergencyLevel.NONE
            detected = []
            confidence = 0.5
            action = "No emergency detected - regular medical consultation if needed"
            reasoning = "No emergency symptoms detected"
        
        return EmergencyDetectionResult(
            level=level,
            detected_conditions=detected,
            confidence=confidence,
            recommended_action=action,
            reasoning=reasoning
        )
    
    def _check_symptoms(
        self,
        text: str,
        symptom_dict: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Check for symptoms in text.
        
        Args:
            text: Text to check
            symptom_dict: Dictionary of symptoms to check
            
        Returns:
            Dictionary of matched symptoms
        """
        matches = {}
        
        for category, patterns in symptom_dict.items():
            for pattern in patterns:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(pattern) + r'\b', text):
                    if category not in matches:
                        matches[category] = []
                    matches[category].append(pattern)
                    break  # One match per category is enough
        
        return matches
    
    def get_response(self) -> str:
        """
        Get emergency response message for last detected emergency.
        
        Returns:
            Emergency response message
        """
        return (
            "🚨 MEDICAL EMERGENCY DETECTED 🚨\n\n"
            "If you are experiencing a medical emergency:\n"
            "• Call 911 immediately (US)\n"
            "• Go to the nearest emergency room\n"
            "• Do not delay seeking professional medical help\n\n"
            "This AI cannot provide emergency medical care."
        )
    
    def categorize(self, text: str) -> str:
        """
        Categorize the type of emergency.
        
        Args:
            text: Text describing symptoms
            
        Returns:
            Category of emergency (cardiac, respiratory, mental_health, general)
        """
        text_lower = text.lower()
        
        # Check for cardiac
        cardiac_terms = ["chest pain", "heart", "cardiac"]
        if any(term in text_lower for term in cardiac_terms):
            return "cardiac"
        
        # Check for respiratory
        respiratory_terms = ["breathe", "breathing", "breath", "respiratory"]
        if any(term in text_lower for term in respiratory_terms):
            return "respiratory"
        
        # Check for mental health
        mental_health_terms = ["suicide", "suicidal", "self-harm", "depression"]
        if any(term in text_lower for term in mental_health_terms):
            return "mental_health"
        
        return "general"
    
    def is_emergency(self, text: str, threshold: EmergencyLevel = EmergencyLevel.URGENT) -> bool:
        """
        Quick check if text describes an emergency.
        
        Args:
            text: Text to check
            threshold: Minimum level to consider emergency
            
        Returns:
            True if emergency detected
        """
        result = self.detect(text)
        
        if threshold == EmergencyLevel.CRITICAL:
            return result.level == EmergencyLevel.CRITICAL
        elif threshold == EmergencyLevel.URGENT:
            return result.level in [EmergencyLevel.CRITICAL, EmergencyLevel.URGENT]
        else:
            return result.level != EmergencyLevel.NONE
    
    def __bool__(self):
        """Allow EmergencyDetectionResult to be used in boolean context."""
        return False  # Default implementation
    
    def get_response(self) -> str:
        """
        Get emergency response message (for backward compatibility).
        
        Returns:
            Emergency guidance message
        """
        if hasattr(self, '_last_result') and self._last_result:
            return self._last_result.recommended_action
        return "Please call 911 or visit the nearest emergency room if you are experiencing a medical emergency."
    
    def categorize(self, text: str) -> str:
        """
        Categorize the type of emergency (for backward compatibility).
        
        Args:
            text: Text to categorize
            
        Returns:
            Category name
        """
        result = self.detect(text)
        self._last_result = result  # Store for get_response
        
        text_lower = text.lower()
        
        # Check categories
        if any(word in text_lower for word in ['chest', 'heart', 'cardiac']):
            return "cardiac"
        elif any(word in text_lower for word in ['breath', 'respiratory', 'lung']):
            return "respiratory"
        elif any(word in text_lower for word in ['suicid', 'kill myself', 'mental']):
            return "mental_health"
        elif any(word in text_lower for word in ['stroke', 'brain', 'neurological']):
            return "neurological"
        else:
            return "general"
    
    def get_emergency_guidance(self, level: EmergencyLevel) -> str:
        """
        Get guidance text for emergency level.
        
        Args:
            level: Emergency level
            
        Returns:
            Guidance text
        """
        if level == EmergencyLevel.CRITICAL:
            return """
🚨 CRITICAL MEDICAL EMERGENCY 🚨

CALL 911 IMMEDIATELY or have someone call while you:
- Stay with the person
- Do not move them unless in immediate danger
- Begin CPR if trained and necessary
- Note time of onset
- Gather medications/medical history if possible

DO NOT:
- Drive yourself to hospital if symptoms affect you
- Wait to see if symptoms improve
- Rely on online medical advice

This is a potentially life-threatening situation requiring immediate professional medical intervention.
"""
        
        elif level == EmergencyLevel.URGENT:
            return """
⚠️ URGENT MEDICAL ATTENTION NEEDED ⚠️

Go to the nearest Emergency Department or call 911 if:
- Symptoms worsen
- New symptoms develop
- You're unsure about safety

Do not wait for a regular appointment. These symptoms may indicate a serious condition requiring prompt evaluation and treatment.

If you're alone and symptoms affect your ability to drive safely, call 911 instead of driving.
"""
        
        elif level == EmergencyLevel.NON_URGENT:
            return """
📞 PROMPT MEDICAL ATTENTION RECOMMENDED

Contact your healthcare provider today or visit an urgent care center. While not immediately life-threatening, these symptoms warrant prompt medical evaluation.

If symptoms suddenly worsen or new concerning symptoms develop, seek emergency care.
"""
        
        else:
            return "No emergency detected. Consult with healthcare provider for ongoing medical concerns."


class PediatricEmergencyDetector(EmergencyDetector):
    """
    Specialized emergency detector for pediatric cases.
    Children have different emergency criteria.
    """
    
    PEDIATRIC_CRITICAL = {
        "infant breathing": ["infant not breathing", "baby not breathing", "blue baby"],
        "infant unresponsive": ["infant unresponsive", "baby won't wake up"],
        "severe dehydration": ["infant dehydration", "no wet diapers", "sunken fontanelle"],
        "infant fever": ["newborn fever", "infant fever over 100.4", "baby fever under 3 months"],
        "seizure": ["infant seizure", "baby convulsions", "febrile seizure prolonged"]
    }
    
    def __init__(self, sensitivity: float = 0.8):
        """Initialize pediatric emergency detector with higher sensitivity."""
        super().__init__(sensitivity)
        # Add pediatric-specific symptoms to critical list
        self.CRITICAL_SYMPTOMS.update(self.PEDIATRIC_CRITICAL)
        
        logger.info("Initialized PediatricEmergencyDetector")


class SuicideRiskDetector:
    """
    Specialized detector for suicide risk.
    Requires special handling and resources.
    """
    
    SUICIDE_INDICATORS = [
        "want to die", "kill myself", "end my life", "suicide",
        "better off dead", "no reason to live", "can't go on",
        "suicide plan", "goodbye forever", "final message"
    ]
    
    SELF_HARM_INDICATORS = [
        "self-harm", "hurt myself", "cutting myself", "self-injury"
    ]
    
    def __init__(self):
        """Initialize suicide risk detector."""
        logger.info("Initialized SuicideRiskDetector")
    
    def detect_risk(self, text: str) -> Tuple[bool, str]:
        """
        Detect suicide risk in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (risk_detected, guidance_message)
        """
        text_lower = text.lower()
        
        # Check for suicide indicators
        suicide_risk = any(indicator in text_lower for indicator in self.SUICIDE_INDICATORS)
        self_harm_risk = any(indicator in text_lower for indicator in self.SELF_HARM_INDICATORS)
        
        if suicide_risk:
            message = """
🆘 SUICIDE PREVENTION RESOURCES 🆘

If you're having thoughts of suicide, please reach out for help immediately:

• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• International: https://findahelpline.com

You are not alone. These services are:
- Free and confidential
- Available 24/7
- Staffed by trained counselors

If you're in immediate danger:
• Call 911
• Go to nearest emergency room
• Stay with someone you trust

Your life matters. Help is available.
"""
            return True, message
        
        elif self_harm_risk:
            message = """
⚠️ SELF-HARM SUPPORT NEEDED ⚠️

If you're struggling with self-harm:

• Crisis Text Line: Text HOME to 741741
• National Suicide Prevention Lifeline: 988
• SAMHSA National Helpline: 1-800-662-4357

Please reach out to:
- Your healthcare provider
- A mental health professional
- A trusted friend or family member
- School counselor (if applicable)

These feelings are treatable. Help is available.
"""
            return True, message
        
        return False, ""
