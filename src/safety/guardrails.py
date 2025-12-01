"""
Main AI safety guardrails system.
Integrates all safety components for comprehensive medical AI safety.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .emergency_detection import EmergencyDetector, EmergencyLevel, SuicideRiskDetector
from .claim_filtering import ProhibitedClaimFilter, MisinformationDetector, ClaimSeverity
from .dosage_validation import DosageValidator, DosageRisk
from .disclaimer_system import DisclaimerSystem, DisclaimerType, DisclaimerConfig
from ..utils import get_logger

logger = get_logger(__name__)


class SafetyLevel(Enum):
    """Overall safety assessment level."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKED = "blocked"


@dataclass
class SafetyCheckResult:
    """Result from safety check."""
    safety_level: SafetyLevel
    is_safe: bool
    emergency_detected: bool
    prohibited_claims: List[str]
    dosage_issues: List[str]
    required_disclaimers: List[DisclaimerType]
    filtered_content: Optional[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, any]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for backward compatibility."""
        return hasattr(self, key)
    
    def get(self, key: str, default=None):
        """Get attribute value (dict-like interface)."""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str):
        """Support dict-like access."""
        return getattr(self, key)
    
    @property
    def is_emergency(self) -> bool:
        """Backward compatible property."""
        return self.emergency_detected


class MedicalAISafetyGuardrails:
    """
    Comprehensive safety guardrails for medical AI.
    Ensures safe, ethical, and compliant AI responses.
    """
    
    def __init__(
        self,
        enable_emergency_detection: bool = True,
        enable_claim_filtering: bool = True,
        enable_dosage_validation: bool = True,
        enable_disclaimers: bool = True,
        strict_mode: bool = True
    ):
        """
        Initialize safety guardrails.
        
        Args:
            enable_emergency_detection: Enable emergency situation detection
            enable_claim_filtering: Enable prohibited claim filtering
            enable_dosage_validation: Enable dosage validation
            enable_disclaimers: Enable automatic disclaimers
            strict_mode: If True, more stringent safety checks
        """
        self.strict_mode = strict_mode
        
        # Initialize components
        if enable_emergency_detection:
            self.emergency_detector = EmergencyDetector()
            self.suicide_detector = SuicideRiskDetector()
        else:
            self.emergency_detector = None
            self.suicide_detector = None
        
        if enable_claim_filtering:
            self.claim_filter = ProhibitedClaimFilter(strict_mode=strict_mode)
            self.misinfo_detector = MisinformationDetector()
        else:
            self.claim_filter = None
            self.misinfo_detector = None
        
        if enable_dosage_validation:
            self.dosage_validator = DosageValidator()
        else:
            self.dosage_validator = None
        
        if enable_disclaimers:
            disclaimer_config = DisclaimerConfig(
                always_add_general=True,
                add_context_specific=True,
                add_limitations=strict_mode
            )
            self.disclaimer_system = DisclaimerSystem(config=disclaimer_config)
        else:
            self.disclaimer_system = None
        
        logger.info(f"Initialized MedicalAISafetyGuardrails (strict_mode={strict_mode})")
    
    def check(self, text: str, check_type: str = "query") -> SafetyCheckResult:
        """
        General safety check method (for backward compatibility).
        
        Args:
            text: Text to check (query or response)
            check_type: Type of check ("query" or "response")
            
        Returns:
            SafetyCheckResult
        """
        if check_type == "response":
            return self.check_response_safety(text)
        else:
            return self.check_query_safety(text)
    
    def check_query_safety(self, query: str) -> SafetyCheckResult:
        """
        Check safety of user query before processing.
        
        Args:
            query: User query/input
            
        Returns:
            SafetyCheckResult
        """
        warnings = []
        recommendations = []
        metadata = {}
        safety_level = SafetyLevel.SAFE
        emergency_detected = False
        prohibited_claims = []
        dosage_issues = []
        required_disclaimers = []
        
        # Check for emergencies
        if self.emergency_detector:
            emergency_result = self.emergency_detector.detect(query)
            metadata["emergency_detection"] = {
                "level": emergency_result.level.value,
                "conditions": emergency_result.detected_conditions,
                "confidence": emergency_result.confidence
            }
            
            if emergency_result.level in [EmergencyLevel.CRITICAL, EmergencyLevel.URGENT]:
                emergency_detected = True
                safety_level = SafetyLevel.CRITICAL
                warnings.append(f"EMERGENCY DETECTED: {emergency_result.reasoning}")
                recommendations.append(emergency_result.recommended_action)
                recommendations.append(self.emergency_detector.get_emergency_guidance(emergency_result.level))
                required_disclaimers.append(DisclaimerType.EMERGENCY)
        
        # Check for suicide risk
        if self.suicide_detector:
            suicide_risk, suicide_message = self.suicide_detector.detect_risk(query)
            if suicide_risk:
                emergency_detected = True
                safety_level = SafetyLevel.CRITICAL
                warnings.append("SUICIDE RISK DETECTED")
                recommendations.append(suicide_message)
                required_disclaimers.append(DisclaimerType.MENTAL_HEALTH)
                metadata["suicide_risk"] = True
        
        # Auto-detect required disclaimers from query
        if self.disclaimer_system:
            detected_disclaimers = self.disclaimer_system.detect_required_disclaimers(query)
            required_disclaimers.extend(detected_disclaimers)
        
        is_safe = safety_level not in [SafetyLevel.CRITICAL, SafetyLevel.BLOCKED]
        
        return SafetyCheckResult(
            safety_level=safety_level,
            is_safe=is_safe,
            emergency_detected=emergency_detected,
            prohibited_claims=prohibited_claims,
            dosage_issues=dosage_issues,
            required_disclaimers=list(set(required_disclaimers)),
            filtered_content=None,
            warnings=warnings,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def check_response_safety(
        self,
        response: str,
        query: Optional[str] = None
    ) -> SafetyCheckResult:
        """
        Check safety of AI response before returning to user.
        
        Args:
            response: AI-generated response
            query: Original user query (for context)
            
        Returns:
            SafetyCheckResult
        """
        warnings = []
        recommendations = []
        metadata = {}
        safety_level = SafetyLevel.SAFE
        emergency_detected = False
        prohibited_claims = []
        dosage_issues = []
        required_disclaimers = []
        filtered_content = None
        
        # Check for prohibited claims
        if self.claim_filter:
            claim_result = self.claim_filter.filter(response, context=query)
            metadata["claim_filtering"] = {
                "severity": claim_result.severity.value,
                "detected": claim_result.detected_claims
            }
            
            if claim_result.severity == ClaimSeverity.DANGEROUS:
                safety_level = SafetyLevel.BLOCKED
                prohibited_claims = claim_result.detected_claims
                warnings.extend(claim_result.warnings)
                filtered_content = claim_result.filtered_content
                recommendations.append(self.claim_filter.get_safety_message(claim_result.severity))
            
            elif claim_result.severity == ClaimSeverity.MISLEADING:
                if safety_level == SafetyLevel.SAFE:
                    safety_level = SafetyLevel.WARNING
                prohibited_claims = claim_result.detected_claims
                warnings.extend(claim_result.warnings)
                filtered_content = claim_result.filtered_content
                recommendations.append(self.claim_filter.get_safety_message(claim_result.severity))
            
            elif claim_result.severity == ClaimSeverity.INAPPROPRIATE:
                if safety_level == SafetyLevel.SAFE:
                    safety_level = SafetyLevel.CAUTION
                warnings.extend(claim_result.warnings)
        
        # Check for misinformation
        if self.misinfo_detector:
            misinfo_detected, misinfo_categories, correction = self.misinfo_detector.detect(response)
            if misinfo_detected:
                metadata["misinformation"] = {
                    "detected": True,
                    "categories": misinfo_categories
                }
                safety_level = SafetyLevel.BLOCKED
                warnings.append("MISINFORMATION DETECTED")
                recommendations.append(correction)
        
        # Parse and validate dosages if mentioned
        if self.dosage_validator:
            dosages = self.dosage_validator.parse_dosage_from_text(response)
            if dosages:
                for dosage in dosages:
                    validation = self.dosage_validator.validate(
                        medication=dosage["medication"],
                        dose=dosage["dose"],
                        unit=dosage["unit"],
                        frequency=dosage.get("frequency")
                    )
                    
                    if validation.risk_level in [DosageRisk.WARNING, DosageRisk.DANGEROUS]:
                        dosage_issues.extend(validation.issues)
                        recommendations.extend(validation.recommendations)
                        
                        if validation.risk_level == DosageRisk.DANGEROUS:
                            if safety_level not in [SafetyLevel.BLOCKED, SafetyLevel.CRITICAL]:
                                safety_level = SafetyLevel.WARNING
                
                metadata["dosage_validation"] = {
                    "dosages_found": len(dosages),
                    "issues": dosage_issues
                }
                
                if dosages:
                    required_disclaimers.append(DisclaimerType.MEDICATION)
        
        # Auto-detect required disclaimers from response
        if self.disclaimer_system:
            detected_disclaimers = self.disclaimer_system.detect_required_disclaimers(response)
            required_disclaimers.extend(detected_disclaimers)
        
        # Determine if safe
        is_safe = safety_level not in [SafetyLevel.BLOCKED]
        
        return SafetyCheckResult(
            safety_level=safety_level,
            is_safe=is_safe,
            emergency_detected=emergency_detected,
            prohibited_claims=prohibited_claims,
            dosage_issues=dosage_issues,
            required_disclaimers=list(set(required_disclaimers)),
            filtered_content=filtered_content,
            warnings=warnings,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def apply_safety_measures(
        self,
        response: str,
        safety_result: SafetyCheckResult
    ) -> str:
        """
        Apply safety measures to response based on safety check.
        
        Args:
            response: Original response
            safety_result: Result from safety check
            
        Returns:
            Modified response with safety measures applied
        """
        # If blocked, return safety message instead
        if safety_result.safety_level == SafetyLevel.BLOCKED:
            blocked_message = """
I apologize, but I cannot provide this information as it may contain:
- Potentially dangerous medical advice
- Misinformation not supported by evidence
- Content that could lead to patient harm

Please consult qualified healthcare professionals for medical advice.

"""
            if safety_result.recommendations:
                blocked_message += "\n" + "\n".join(safety_result.recommendations)
            
            return blocked_message
        
        # Use filtered content if available
        modified_response = safety_result.filtered_content or response
        
        # Add warnings
        if safety_result.warnings:
            warning_text = "\n\n⚠️ SAFETY WARNINGS ⚠️\n" + "\n".join(f"• {w}" for w in safety_result.warnings)
            modified_response = warning_text + "\n\n" + modified_response
        
        # Add recommendations
        if safety_result.recommendations:
            modified_response += "\n\n" + "\n".join(safety_result.recommendations)
        
        # Add disclaimers
        if self.disclaimer_system and safety_result.required_disclaimers:
            modified_response = self.disclaimer_system.add_disclaimers(
                modified_response,
                safety_result.required_disclaimers
            )
        
        return modified_response
    
    def safe_response(
        self,
        query: str,
        response: str
    ) -> Tuple[bool, str, SafetyCheckResult]:
        """
        Complete safety pipeline: check query, check response, apply measures.
        
        Args:
            query: User query
            response: AI response
            
        Returns:
            Tuple of (is_safe, modified_response, safety_result)
        """
        # Check query safety first
        query_safety = self.check_query_safety(query)
        
        # If critical query (emergency), prioritize that
        if query_safety.emergency_detected:
            emergency_response = "\n".join(query_safety.recommendations)
            return False, emergency_response, query_safety
        
        # Check response safety
        response_safety = self.check_response_safety(response, query)
        
        # Merge required disclaimers
        all_disclaimers = list(set(
            query_safety.required_disclaimers + 
            response_safety.required_disclaimers
        ))
        response_safety.required_disclaimers = all_disclaimers
        
        # Apply safety measures
        safe_response = self.apply_safety_measures(response, response_safety)
        
        return response_safety.is_safe, safe_response, response_safety
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety system statistics."""
        stats = {
            "strict_mode": self.strict_mode,
            "components_enabled": {
                "emergency_detection": self.emergency_detector is not None,
                "claim_filtering": self.claim_filter is not None,
                "dosage_validation": self.dosage_validator is not None,
                "disclaimers": self.disclaimer_system is not None
            }
        }
        return stats


# Convenience function for quick safety check
def quick_safety_check(text: str, is_response: bool = False) -> Tuple[bool, List[str]]:
    """
    Quick safety check without full guardrails initialization.
    
    Args:
        text: Text to check
        is_response: Whether text is AI response (vs user query)
        
    Returns:
        Tuple of (is_safe, warnings)
    """
    guardrails = MedicalAISafetyGuardrails()
    
    if is_response:
        result = guardrails.check_response_safety(text)
    else:
        result = guardrails.check_query_safety(text)
    
    return result.is_safe, result.warnings
