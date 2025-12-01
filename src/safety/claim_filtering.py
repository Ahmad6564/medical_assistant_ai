"""
Filter prohibited medical claims and misinformation.
Prevents generation of dangerous or unsubstantiated medical claims.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)


class ClaimSeverity(Enum):
    """Severity of prohibited claim."""
    DANGEROUS = "dangerous"  # Potentially harmful
    MISLEADING = "misleading"  # Unsupported by evidence
    INAPPROPRIATE = "inappropriate"  # Outside AI scope
    ACCEPTABLE = "acceptable"  # No issues detected


@dataclass
class ClaimFilterResult:
    """Result from claim filtering."""
    severity: ClaimSeverity
    detected_claims: List[str]
    filtered_content: Optional[str]
    warnings: List[str]
    requires_disclaimer: bool


class ProhibitedClaimFilter:
    """
    Filter prohibited and dangerous medical claims.
    Prevents AI from making dangerous recommendations.
    """
    
    # Prohibited cure claims
    PROHIBITED_CURE_CLAIMS = [
        "cure cancer", "cure diabetes", "cure hiv", "cure aids",
        "cure alzheimer", "cure autism", "cure depression",
        "miracle cure", "guaranteed cure", "permanent cure"
    ]
    
    # Prohibited diagnostic claims
    PROHIBITED_DIAGNOSTIC_CLAIMS = [
        "you have", "you definitely have", "diagnosis is",
        "you are diagnosed with", "confirmed diagnosis",
        "definitely suffering from"
    ]
    
    # Dangerous advice patterns
    DANGEROUS_ADVICE = [
        "don't see a doctor", "avoid doctors", "don't go to hospital",
        "skip medication", "stop taking", "don't take medication",
        "replace medication with", "instead of medication",
        "medical treatment is unnecessary"
    ]
    
    # Unproven treatment claims
    UNPROVEN_TREATMENTS = [
        "essential oils cure", "homeopathy cures", "crystals heal",
        "detox cleanse", "juice cleanse cures",
        "magnetic therapy", "miracle supplement"
    ]
    
    # Off-label use without disclaimer
    OFF_LABEL_PATTERNS = [
        r"use .+ for .+ \(not approved\)",
        r"off-label use of",
        r"unapproved use"
    ]
    
    # Dosage modification advice
    DOSAGE_MODIFICATION = [
        "increase your dose", "decrease your dose", "double your dose",
        "stop taking suddenly", "change your dosage"
    ]
    
    # Self-treatment of serious conditions
    SERIOUS_SELF_TREATMENT = [
        "treat cancer at home", "self-treat heart disease",
        "manage stroke yourself", "self-treat diabetes"
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize claim filter.
        
        Args:
            strict_mode: If True, more aggressive filtering
        """
        self.strict_mode = strict_mode
        logger.info(f"Initialized ProhibitedClaimFilter (strict_mode={strict_mode})")
    
    def is_claim(self, text: str) -> bool:
        """
        Check if text contains a medical claim.
        
        Args:
            text: Text to check
            
        Returns:
            True if claim detected
        """
        text_lower = text.lower()
        
        # Check for cure claims (including partial matches like "cures")
        cure_patterns = ["cure", "cures", "cured", "curing"]
        for cure_word in cure_patterns:
            if cure_word in text_lower:
                # Check if it's used in a claim context (not "can be cured with medical treatment")
                if any(term in text_lower for term in ["this", "will", "drug", "treatment"]):
                    return True
        
        # Check for prohibited cure claims
        if any(claim in text_lower for claim in self.PROHIBITED_CURE_CLAIMS):
            return True
        
        # Check for diagnostic claims
        if any(claim in text_lower for claim in self.PROHIBITED_DIAGNOSTIC_CLAIMS):
            return True
        
        # Check for directive language (should, will, definitely)
        directive_patterns = ["should take", "will cure", "will definitely", "definitely work"]
        if any(pattern in text_lower for pattern in directive_patterns):
            return True
        
        return False
    
    def add_disclaimer(self, text: str) -> str:
        """
        Add appropriate disclaimer to text.
        
        Args:
            text: Text to add disclaimer to
            
        Returns:
            Text with disclaimer
        """
        disclaimer = (
            "\n\n⚠️ DISCLAIMER: This information is for educational purposes only. "
            "Always consult with a qualified healthcare provider before making any medical decisions."
        )
        return text + disclaimer
    
    def get_claim_score(self, text: str) -> float:
        """
        Get confidence score for claim detection.
        
        Args:
            text: Text to score
            
        Returns:
            Score from 0 to 1 (higher = more likely to be a claim)
        """
        text_lower = text.lower()
        score = 0.0
        
        # Strong claim indicators
        strong_indicators = ["cure", "will", "definitely", "should", "must", "guaranteed"]
        for indicator in strong_indicators:
            if indicator in text_lower:
                score += 0.3
        
        # Medical directive language
        if any(claim in text_lower for claim in self.PROHIBITED_DIAGNOSTIC_CLAIMS):
            score += 0.4
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def filter(self, text: str, context: Optional[str] = None) -> ClaimFilterResult:
        """
        Filter prohibited claims from text.
        
        Args:
            text: Text to filter
            context: Additional context (e.g., user query)
            
        Returns:
            ClaimFilterResult
        """
        text_lower = text.lower()
        detected_claims = []
        warnings = []
        severity = ClaimSeverity.ACCEPTABLE
        
        # Check for dangerous claims
        dangerous = self._check_patterns(text_lower, [
            *self.DANGEROUS_ADVICE,
            *self.SERIOUS_SELF_TREATMENT,
            *self.PROHIBITED_CURE_CLAIMS
        ])
        
        if dangerous:
            detected_claims.extend(dangerous)
            severity = ClaimSeverity.DANGEROUS
            warnings.append("Detected potentially dangerous medical advice")
        
        # Check for misleading claims
        misleading = self._check_patterns(text_lower, [
            *self.UNPROVEN_TREATMENTS,
            *self.PROHIBITED_DIAGNOSTIC_CLAIMS
        ])
        
        if misleading:
            detected_claims.extend(misleading)
            if severity != ClaimSeverity.DANGEROUS:
                severity = ClaimSeverity.MISLEADING
            warnings.append("Detected misleading or unsupported claims")
        
        # Check for dosage modification advice
        dosage_issues = self._check_patterns(text_lower, self.DOSAGE_MODIFICATION)
        if dosage_issues:
            detected_claims.extend(dosage_issues)
            if severity == ClaimSeverity.ACCEPTABLE:
                severity = ClaimSeverity.INAPPROPRIATE
            warnings.append("Detected medication dosage modification advice")
        
        # Filter content if needed
        filtered_content = None
        if severity in [ClaimSeverity.DANGEROUS, ClaimSeverity.MISLEADING]:
            filtered_content = self._apply_filter(text, detected_claims)
        
        # Determine if disclaimer needed
        requires_disclaimer = (
            severity != ClaimSeverity.ACCEPTABLE or
            self._mentions_treatment(text_lower) or
            self._mentions_diagnosis(text_lower)
        )
        
        return ClaimFilterResult(
            severity=severity,
            detected_claims=detected_claims,
            filtered_content=filtered_content,
            warnings=warnings,
            requires_disclaimer=requires_disclaimer
        )
    
    def _check_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """
        Check for patterns in text.
        
        Args:
            text: Text to check
            patterns: Patterns to look for
            
        Returns:
            List of matched patterns
        """
        matches = []
        for pattern in patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text):
                matches.append(pattern)
        return matches
    
    def _apply_filter(self, text: str, prohibited_claims: List[str]) -> str:
        """
        Apply filtering to remove/modify prohibited claims.
        
        Args:
            text: Original text
            prohibited_claims: List of prohibited claims found
            
        Returns:
            Filtered text
        """
        filtered = text
        
        # Replace prohibited phrases with safer alternatives
        replacements = {
            "cure": "help manage",
            "guaranteed": "may help",
            "definitely": "possibly",
            "you have": "symptoms suggest possible",
            "diagnosis is": "possible diagnosis includes"
        }
        
        for original, replacement in replacements.items():
            filtered = re.sub(
                r'\b' + original + r'\b',
                replacement,
                filtered,
                flags=re.IGNORECASE
            )
        
        return filtered
    
    def _mentions_treatment(self, text: str) -> bool:
        """Check if text mentions treatments."""
        treatment_keywords = [
            "treatment", "therapy", "medication", "drug", "prescription",
            "dose", "dosage", "take", "administer"
        ]
        return any(keyword in text for keyword in treatment_keywords)
    
    def _mentions_diagnosis(self, text: str) -> bool:
        """Check if text mentions diagnoses."""
        diagnosis_keywords = [
            "diagnosis", "condition", "disease", "disorder", "syndrome",
            "you have", "suffering from", "diagnosed with"
        ]
        return any(keyword in text for keyword in diagnosis_keywords)
    
    def get_safety_message(self, severity: ClaimSeverity) -> str:
        """
        Get safety message for claim severity.
        
        Args:
            severity: Claim severity level
            
        Returns:
            Safety message
        """
        if severity == ClaimSeverity.DANGEROUS:
            return """
⛔ SAFETY WARNING ⛔

This response contained potentially dangerous medical advice that has been filtered.

NEVER:
- Stop prescribed medications without consulting your doctor
- Attempt to self-treat serious medical conditions
- Rely on unproven treatments for serious illnesses
- Avoid seeking professional medical care

Always consult qualified healthcare professionals for medical advice and treatment decisions.
"""
        
        elif severity == ClaimSeverity.MISLEADING:
            return """
⚠️ INFORMATION QUALITY NOTICE ⚠️

Some content has been filtered as it contained claims not supported by scientific evidence.

Remember:
- Rely on evidence-based medical information
- Be skeptical of "miracle cures" or guaranteed results
- Verify medical information with healthcare professionals
- Consider multiple reputable sources
"""
        
        elif severity == ClaimSeverity.INAPPROPRIATE:
            return """
ℹ️ DISCLAIMER ℹ️

This information is for educational purposes only. 

DO NOT make medication changes without consulting your healthcare provider.
Dosage modifications should only be made under medical supervision.
"""
        
        return ""


class MisinformationDetector:
    """
    Detect common medical misinformation patterns.
    """
    
    MISINFORMATION_PATTERNS = {
        "vaccine_misinfo": [
            "vaccines cause autism", "vaccines are dangerous",
            "skip vaccinations", "vaccines have microchips"
        ],
        "covid_misinfo": [
            "covid is a hoax", "masks don't work", "ivermectin cures covid",
            "bleach cures covid", "covid is just the flu"
        ],
        "alternative_medicine_misinfo": [
            "chiropractic cures all", "acupuncture cures cancer",
            "herbs can replace insulin", "natural remedies cure everything"
        ],
        "conspiracy": [
            "doctors hiding cure", "big pharma conspiracy",
            "government poisoning", "chemtrails causing illness"
        ]
    }
    
    def __init__(self):
        """Initialize misinformation detector."""
        logger.info("Initialized MisinformationDetector")
    
    def detect(self, text: str) -> Tuple[bool, List[str], str]:
        """
        Detect misinformation in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (detected, categories, correction_message)
        """
        text_lower = text.lower()
        detected_categories = []
        
        for category, patterns in self.MISINFORMATION_PATTERNS.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_categories.append(category)
        
        if detected_categories:
            correction = self._get_correction(detected_categories)
            return True, detected_categories, correction
        
        return False, [], ""
    
    def _get_correction(self, categories: List[str]) -> str:
        """Get correction message for detected misinformation."""
        corrections = {
            "vaccine_misinfo": """
VACCINE SAFETY INFORMATION:
- Vaccines are safe and effective, backed by extensive scientific evidence
- Vaccines do NOT cause autism (thoroughly debunked)
- Vaccines prevent serious diseases and save lives
- Consult CDC, WHO, or healthcare providers for accurate vaccine information
""",
            "covid_misinfo": """
COVID-19 FACTS:
- COVID-19 is a real disease caused by SARS-CoV-2 virus
- Masks are effective at reducing transmission
- Vaccines are safe and highly effective
- Consult CDC or WHO for accurate COVID-19 information
- Use only FDA-approved or EUA treatments
""",
            "alternative_medicine_misinfo": """
COMPLEMENTARY MEDICINE NOTE:
- Some complementary therapies may help alongside conventional treatment
- They should NOT replace evidence-based medical care for serious conditions
- Always inform your healthcare provider about all treatments used
- Be skeptical of cure-all claims
""",
            "conspiracy": """
EVIDENCE-BASED MEDICINE:
- Medical treatments are based on scientific research and clinical trials
- Healthcare professionals follow evidence-based guidelines
- Conspiracy theories are not supported by scientific evidence
- Trust reputable sources: medical journals, CDC, WHO, major medical institutions
"""
        }
        
        messages = [corrections.get(cat, "") for cat in categories if cat in corrections]
        return "\n".join(messages)
