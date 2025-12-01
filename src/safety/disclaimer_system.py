"""
Medical disclaimer system.
Automatically adds appropriate disclaimers to medical AI responses.
"""

from typing import Optional, List
from enum import Enum
from dataclasses import dataclass

from ..utils import get_logger

logger = get_logger(__name__)


class DisclaimerType(Enum):
    """Types of disclaimers."""
    GENERAL = "general"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    EMERGENCY = "emergency"
    PEDIATRIC = "pediatric"
    MENTAL_HEALTH = "mental_health"
    PREGNANCY = "pregnancy"


@dataclass
class DisclaimerConfig:
    """Configuration for disclaimer system."""
    always_add_general: bool = True
    add_context_specific: bool = True
    add_emergency_warning: bool = True
    add_limitations: bool = True
    placement: str = "end"  # "start" or "end"


class DisclaimerSystem:
    """
    Manage medical disclaimers for AI responses.
    Ensures appropriate warnings and legal protection.
    """
    
    DISCLAIMERS = {
        DisclaimerType.GENERAL: """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚕️ IMPORTANT MEDICAL DISCLAIMER ⚕️

This information is provided for educational purposes only and should not be 
considered medical advice, diagnosis, or treatment recommendations.

Key Points:
• Always consult qualified healthcare professionals (doctor) for medical decisions
• Individual cases vary significantly - personalized evaluation is essential  
• This AI assistant cannot replace clinical judgment or examination
• Emergency situations require immediate professional medical attention
• Information may not reflect the most current medical research or guidelines

By using this service, you acknowledge that:
- You understand this is not a substitute for professional medical care
- You will seek appropriate medical attention for health concerns
- You will not rely solely on this information for health decisions

© 2024 Medical AI Assistant - For Educational Use Only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        
        DisclaimerType.DIAGNOSIS: """
⚠️ DIAGNOSTIC INFORMATION DISCLAIMER ⚠️

The information provided discusses possible diagnoses for educational purposes.

Important Limitations:
• AI cannot perform physical examinations or diagnostic tests
• Symptoms may indicate multiple conditions
• Definitive diagnosis requires clinical evaluation
• Self-diagnosis can be dangerous and is not recommended

Required Actions:
→ Consult a healthcare provider for proper diagnostic evaluation
→ Do not delay seeking care based on AI information
→ Provide your healthcare provider with complete symptom information
""",
        
        DisclaimerType.TREATMENT: """
⚠️ TREATMENT INFORMATION DISCLAIMER ⚠️

Treatment information is provided for educational awareness only.

Critical Considerations:
• Treatment decisions must be individualized to each patient
• Side effects, contraindications, and drug interactions vary
• Alternative treatments may be more appropriate for your situation
• Treatment should only begin under medical supervision

Required Actions:
→ Discuss treatment options with your healthcare provider
→ Never start, stop, or modify treatments without medical guidance
→ Inform your provider of all medications and supplements you take
→ Report any adverse effects to your healthcare provider immediately
""",
        
        DisclaimerType.MEDICATION: """
💊 MEDICATION INFORMATION DISCLAIMER 💊

Medication information is for educational purposes only.

NEVER:
• Start new medications without a prescription
• Adjust doses without consulting your healthcare provider
• Stop prescribed medications abruptly
• Use others' medications or share your medications

ALWAYS:
• Follow prescribing information and provider instructions
• Report side effects to your healthcare provider
• Inform providers of all medications, supplements, and allergies
• Ask your pharmacist if you have questions about your medications

In case of severe side effects or allergic reactions, seek emergency care immediately.
""",
        
        DisclaimerType.EMERGENCY: """
🚨 EMERGENCY SITUATIONS DISCLAIMER 🚨

If you are experiencing a medical emergency:

CALL 911 IMMEDIATELY or go to the nearest emergency room

Emergency Warning Signs Include:
• Chest pain or pressure
• Difficulty breathing
• Uncontrolled bleeding
• Loss of consciousness
• Severe head injury
• Stroke symptoms (facial drooping, arm weakness, speech difficulty)
• Severe allergic reaction
• Suicidal thoughts or self-harm

DO NOT rely on online information in emergency situations.
Minutes matter in emergencies - seek immediate professional help.
""",
        
        DisclaimerType.PEDIATRIC: """
👶 PEDIATRIC INFORMATION DISCLAIMER 👶

Information about children's health requires special caution.

Important Considerations:
• Children are not small adults - they require specialized care
• Dosing calculations differ significantly for pediatric patients
• Some medications are contraindicated in children
• Growth and development require ongoing monitoring

Required Actions:
→ Always consult a pediatrician or pediatric specialist
→ Never give children adult medications without medical guidance
→ Report any concerning symptoms to your child's healthcare provider
→ Keep poison control number readily available: 1-800-222-1222

Infants under 3 months with fever require immediate medical evaluation.
""",
        
        DisclaimerType.MENTAL_HEALTH: """
🧠 MENTAL HEALTH INFORMATION DISCLAIMER 🧠

Mental health information is provided for educational awareness.

Important Resources:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• SAMHSA National Helpline: 1-800-662-4357

Key Points:
• Mental health conditions are medical conditions requiring professional care
• Effective treatments are available - recovery is possible
• Medication decisions should be made with psychiatric providers
• Therapy and counseling are important treatment components

If you're experiencing a mental health crisis, seek immediate help.
You are not alone - help is available 24/7.
""",
        
        DisclaimerType.PREGNANCY: """
🤰 PREGNANCY & LACTATION DISCLAIMER 🤰

Information regarding pregnancy and breastfeeding requires special consideration.

Critical Considerations:
• Many medications affect pregnancy and fetal development
• Risk categories help guide but don't replace clinical judgment
• Herbal supplements and "natural" products can be harmful
• Lactation safety varies by medication

Required Actions:
→ Consult your OB/GYN or healthcare provider before:
  - Taking any new medications or supplements
  - Stopping current medications
  - Using herbal or alternative remedies

→ Inform ALL healthcare providers that you are pregnant or breastfeeding

Emergency: Pregnancy-related emergencies (bleeding, severe pain, decreased fetal 
movement) require immediate medical attention.
"""
    }
    
    LIMITATION_NOTICES = """
ℹ️ AI SYSTEM LIMITATIONS ℹ️

This AI assistant has important limitations:

What This System CANNOT Do:
✗ Perform physical examinations
✗ Order or interpret diagnostic tests
✗ Prescribe medications
✗ Provide legally binding medical advice
✗ Replace in-person medical evaluation
✗ Account for your complete medical history
✗ Make definitive diagnoses
✗ Guarantee accuracy of all information

What This System CAN Do:
✓ Provide general medical education
✓ Explain medical concepts
✓ Summarize research and guidelines
✓ Help you prepare questions for your doctor
✓ Offer information for informed health discussions

Always verify critical health information with healthcare professionals.
"""
    
    def __init__(self, config: Optional[DisclaimerConfig] = None):
        """
        Initialize disclaimer system.
        
        Args:
            config: Disclaimer configuration
        """
        self.config = config or DisclaimerConfig()
        logger.info("Initialized DisclaimerSystem")
    
    def add_disclaimers(
        self,
        text: str,
        disclaimer_types: List[DisclaimerType],
        add_limitations: Optional[bool] = None
    ) -> str:
        """
        Add appropriate disclaimers to text.
        
        Args:
            text: Original text
            disclaimer_types: Types of disclaimers to add
            add_limitations: Override config for limitations
            
        Returns:
            Text with disclaimers
        """
        disclaimers = []
        
        # Add general disclaimer if configured
        if self.config.always_add_general and DisclaimerType.GENERAL not in disclaimer_types:
            disclaimer_types = [DisclaimerType.GENERAL] + disclaimer_types
        
        # Collect disclaimer texts
        for dtype in disclaimer_types:
            if dtype in self.DISCLAIMERS:
                disclaimers.append(self.DISCLAIMERS[dtype])
        
        # Add limitations if configured
        if (add_limitations if add_limitations is not None else self.config.add_limitations):
            disclaimers.append(self.LIMITATION_NOTICES)
        
        # Combine disclaimers
        disclaimer_text = "\n\n".join(disclaimers)
        
        # Add to text based on placement
        if self.config.placement == "start":
            return disclaimer_text + "\n\n" + text
        else:
            return text + "\n\n" + disclaimer_text
    
    def get_quick_disclaimer(self, disclaimer_type: DisclaimerType) -> str:
        """
        Get a single disclaimer text.
        
        Args:
            disclaimer_type: Type of disclaimer
            
        Returns:
            Disclaimer text
        """
        return self.DISCLAIMERS.get(disclaimer_type, self.DISCLAIMERS[DisclaimerType.GENERAL])
    
    def get_disclaimer(self, disclaimer_type: DisclaimerType) -> str:
        """
        Get a single disclaimer text (alias for get_quick_disclaimer).
        
        Args:
            disclaimer_type: Type of disclaimer
            
        Returns:
            Disclaimer text
        """
        return self.get_quick_disclaimer(disclaimer_type)
    
    def detect_required_disclaimers(self, text: str) -> List[DisclaimerType]:
        """
        Automatically detect which disclaimers are needed based on content.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of required disclaimer types
        """
        text_lower = text.lower()
        required = []
        
        # Check for diagnosis-related content
        diagnosis_keywords = [
            "diagnosis", "diagnosed with", "condition", "disease",
            "disorder", "syndrome", "may have", "could be"
        ]
        if any(keyword in text_lower for keyword in diagnosis_keywords):
            required.append(DisclaimerType.DIAGNOSIS)
        
        # Check for treatment-related content
        treatment_keywords = [
            "treatment", "therapy", "manage", "procedure",
            "surgery", "intervention"
        ]
        if any(keyword in text_lower for keyword in treatment_keywords):
            required.append(DisclaimerType.TREATMENT)
        
        # Check for medication-related content
        medication_keywords = [
            "medication", "drug", "prescription", "dose", "dosage",
            "mg", "mcg", "pill", "tablet", "capsule"
        ]
        if any(keyword in text_lower for keyword in medication_keywords):
            required.append(DisclaimerType.MEDICATION)
        
        # Check for emergency content
        emergency_keywords = [
            "emergency", "urgent", "911", "call ambulance",
            "life-threatening", "critical"
        ]
        if any(keyword in text_lower for keyword in emergency_keywords):
            required.append(DisclaimerType.EMERGENCY)
        
        # Check for pediatric content
        pediatric_keywords = [
            "child", "children", "pediatric", "infant", "baby",
            "toddler", "adolescent"
        ]
        if any(keyword in text_lower for keyword in pediatric_keywords):
            required.append(DisclaimerType.PEDIATRIC)
        
        # Check for mental health content
        mental_health_keywords = [
            "depression", "anxiety", "mental health", "psychiatric",
            "suicide", "self-harm", "ptsd", "bipolar"
        ]
        if any(keyword in text_lower for keyword in mental_health_keywords):
            required.append(DisclaimerType.MENTAL_HEALTH)
        
        # Check for pregnancy content
        pregnancy_keywords = [
            "pregnancy", "pregnant", "breastfeeding", "nursing",
            "lactation", "prenatal", "postpartum"
        ]
        if any(keyword in text_lower for keyword in pregnancy_keywords):
            required.append(DisclaimerType.PREGNANCY)
        
        return required


# Pre-formatted short disclaimers for inline use
SHORT_DISCLAIMERS = {
    "general": "⚠️ This is not medical advice. Consult a healthcare professional.",
    "diagnosis": "⚠️ Cannot diagnose. See a doctor for proper evaluation.",
    "treatment": "⚠️ Discuss treatment options with your healthcare provider.",
    "medication": "💊 Never change medications without consulting your doctor.",
    "emergency": "🚨 If emergency, call 911 immediately.",
    "pediatric": "👶 Always consult a pediatrician for children's health.",
    "mental_health": "🧠 Mental health crisis? Call 988 (Suicide Prevention Lifeline).",
    "pregnancy": "🤰 Consult your OB/GYN before taking any medications during pregnancy."
}
