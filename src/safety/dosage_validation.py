"""
Dosage validation system for medication safety.
Validates and checks medication dosages for safety.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)


class DosageRisk(Enum):
    """Risk level for dosage."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGEROUS = "dangerous"


@dataclass
class DosageValidationResult:
    """Result from dosage validation."""
    risk_level: DosageRisk
    issues: List[str]
    recommendations: List[str]
    requires_review: bool


class DosageValidator:
    """
    Validate medication dosages for safety.
    Note: This is a basic validator. Clinical decision support systems
    should integrate with comprehensive drug databases.
    """
    
    # Common medication dosage ranges (mg/day for adults)
    # WARNING: These are simplified examples. Real systems need comprehensive drug databases
    DOSAGE_RANGES = {
        "metformin": {"min": 500, "max": 2550, "unit": "mg/day"},
        "lisinopril": {"min": 2.5, "max": 40, "unit": "mg/day"},
        "atorvastatin": {"min": 10, "max": 80, "unit": "mg/day"},
        "amlodipine": {"min": 2.5, "max": 10, "unit": "mg/day"},
        "levothyroxine": {"min": 25, "max": 300, "unit": "mcg/day"},
        "omeprazole": {"min": 20, "max": 40, "unit": "mg/day"},
        "sertraline": {"min": 25, "max": 200, "unit": "mg/day"},
        "gabapentin": {"min": 300, "max": 3600, "unit": "mg/day"},
        "ibuprofen": {"min": 200, "max": 2400, "unit": "mg/day"},
        "acetaminophen": {"min": 325, "max": 4000, "unit": "mg/day"},
        "aspirin": {"min": 81, "max": 325, "unit": "mg/day"},
        "insulin": {"min": 0.2, "max": 2.0, "unit": "units/kg/day"}
    }
    
    # Pediatric warning medications (require special dosing)
    PEDIATRIC_CAUTION = [
        "aspirin", "codeine", "tramadol", "promethazine",
        "diphenhydramine"
    ]
    
    # Geriatric caution medications (Beers Criteria subset)
    GERIATRIC_CAUTION = [
        "benzodiazepines", "anticholinergics", "nsaids",
        "diphenhydramine", "hydroxyzine"
    ]
    
    # Narrow therapeutic index drugs (require careful monitoring)
    NARROW_THERAPEUTIC_INDEX = [
        "warfarin", "digoxin", "lithium", "theophylline",
        "phenytoin", "levothyroxine"
    ]
    
    def __init__(self):
        """Initialize dosage validator."""
        logger.info("Initialized DosageValidator")
    
    def validate(
        self,
        medication: str,
        dose: float,
        unit: str,
        frequency: Optional[str] = None,
        patient_age: Optional[int] = None,
        patient_weight: Optional[float] = None
    ) -> DosageValidationResult:
        """
        Validate medication dosage.
        
        Args:
            medication: Medication name
            dose: Dose amount
            unit: Dose unit (mg, mcg, etc.)
            frequency: Dosing frequency
            patient_age: Patient age in years
            patient_weight: Patient weight in kg
            
        Returns:
            DosageValidationResult
        """
        issues = []
        recommendations = []
        risk_level = DosageRisk.SAFE
        
        medication_lower = medication.lower()
        
        # Check against known ranges
        if medication_lower in self.DOSAGE_RANGES:
            range_info = self.DOSAGE_RANGES[medication_lower]
            
            # Calculate daily dose
            daily_dose = self._calculate_daily_dose(dose, frequency, unit)
            
            if daily_dose is not None:
                # Check if within range
                if daily_dose < range_info["min"]:
                    issues.append(f"Dose below typical range ({range_info['min']} {range_info['unit']})")
                    recommendations.append("Verify if subtherapeutic dose is intentional")
                    risk_level = DosageRisk.CAUTION
                
                elif daily_dose > range_info["max"]:
                    issues.append(f"Dose exceeds typical maximum ({range_info['max']} {range_info['unit']})")
                    recommendations.append("Review for potential overdose risk")
                    risk_level = DosageRisk.DANGEROUS
                
                elif daily_dose > range_info["max"] * 0.8:
                    issues.append(f"Dose near maximum limit")
                    recommendations.append("Monitor for side effects")
                    risk_level = DosageRisk.WARNING
        
        # Special population checks
        if patient_age is not None:
            # Pediatric checks
            if patient_age < 18:
                if medication_lower in self.PEDIATRIC_CAUTION:
                    issues.append(f"{medication} requires special caution in pediatric patients")
                    recommendations.append("Verify pediatric dosing guidelines")
                    if risk_level == DosageRisk.SAFE:
                        risk_level = DosageRisk.CAUTION
            
            # Geriatric checks
            if patient_age >= 65:
                if any(med in medication_lower for med in self.GERIATRIC_CAUTION):
                    issues.append(f"{medication} requires caution in elderly patients")
                    recommendations.append("Consider lower dose or alternative medication")
                    if risk_level == DosageRisk.SAFE:
                        risk_level = DosageRisk.CAUTION
        
        # Narrow therapeutic index check
        if any(nti in medication_lower for nti in self.NARROW_THERAPEUTIC_INDEX):
            recommendations.append(f"{medication} has narrow therapeutic index - requires careful monitoring")
            recommendations.append("Regular serum level monitoring recommended")
        
        # General safety recommendations
        if not issues:
            recommendations.append("Dose appears within typical range")
            recommendations.append("Always confirm with current prescribing information")
        
        requires_review = risk_level in [DosageRisk.WARNING, DosageRisk.DANGEROUS]
        
        return DosageValidationResult(
            risk_level=risk_level,
            issues=issues,
            recommendations=recommendations,
            requires_review=requires_review
        )
    
    def _calculate_daily_dose(
        self,
        dose: float,
        frequency: Optional[str],
        unit: str
    ) -> Optional[float]:
        """
        Calculate total daily dose.
        
        Args:
            dose: Single dose amount
            frequency: Dosing frequency
            unit: Dose unit
            
        Returns:
            Daily dose in standard units
        """
        if frequency is None:
            return dose
        
        frequency_lower = frequency.lower()
        
        # Parse frequency
        multiplier = 1
        if "once" in frequency_lower or "daily" in frequency_lower or "qd" in frequency_lower:
            multiplier = 1
        elif "twice" in frequency_lower or "bid" in frequency_lower or "b.i.d" in frequency_lower:
            multiplier = 2
        elif "three" in frequency_lower or "tid" in frequency_lower or "t.i.d" in frequency_lower:
            multiplier = 3
        elif "four" in frequency_lower or "qid" in frequency_lower or "q.i.d" in frequency_lower:
            multiplier = 4
        elif "every" in frequency_lower:
            # Extract hours (e.g., "every 8 hours")
            match = re.search(r'every (\d+) hour', frequency_lower)
            if match:
                hours = int(match.group(1))
                multiplier = 24 / hours
        
        daily_dose = dose * multiplier
        
        # Convert units if needed
        if unit.lower() in ["mcg", "µg", "ug"]:
            daily_dose = daily_dose / 1000  # Convert to mg
        elif unit.lower() in ["g", "gm", "gram"]:
            daily_dose = daily_dose * 1000  # Convert to mg
        
        return daily_dose
    
    def check_drug_interaction(
        self,
        medications: List[str]
    ) -> Dict[str, List[str]]:
        """
        Basic drug interaction checker.
        Note: Real systems need comprehensive interaction databases.
        
        Args:
            medications: List of medication names
            
        Returns:
            Dictionary of potential interactions
        """
        interactions = {}
        medications_lower = [med.lower() for med in medications]
        
        # Common interaction pairs (simplified examples)
        known_interactions = {
            ("warfarin", "aspirin"): "Increased bleeding risk",
            ("warfarin", "nsaid"): "Increased bleeding risk",
            ("ace inhibitor", "potassium"): "Hyperkalemia risk",
            ("ssri", "nsaid"): "Increased GI bleeding risk",
            ("maoi", "ssri"): "Serotonin syndrome risk - DANGEROUS",
            ("benzodiazepine", "opioid"): "Respiratory depression risk - CAUTION"
        }
        
        for med1 in medications_lower:
            for med2 in medications_lower:
                if med1 != med2:
                    for (drug1, drug2), interaction in known_interactions.items():
                        if (drug1 in med1 and drug2 in med2) or (drug2 in med1 and drug1 in med2):
                            key = f"{med1} + {med2}"
                            if key not in interactions:
                                interactions[key] = []
                            interactions[key].append(interaction)
        
        return interactions
    
    def extract_dosage(self, text: str) -> Dict[str, any]:
        """
        Extract dosage information from text (for backward compatibility).
        
        Args:
            text: Text containing dosage information
            
        Returns:
            Dosage information dictionary
        """
        dosages = self.parse_dosage_from_text(text)
        if dosages:
            return dosages[0]
        # Return with aliases for test compatibility
        return {
            "medication": None,
            "dose": None,
            "unit": None,
            "frequency": None,
            "drug": None,
            "amount": None
        }
    
    def get_warning(self, medication: str, dose: float, unit: str) -> str:
        """
        Get warning for a medication dosage (for backward compatibility).
        
        Args:
            medication: Medication name
            dose: Dose amount
            unit: Dose unit
            
        Returns:
            Warning message
        """
        result = self.validate(medication, dose, unit)
        
        if result.risk_level == DosageRisk.DANGEROUS:
            return f"⚠️ DANGEROUS dosage detected for {medication}. Please consult a healthcare provider immediately."
        elif result.risk_level == DosageRisk.WARNING:
            return f"⚠️ WARNING: Dosage for {medication} requires careful monitoring."
        elif result.risk_level == DosageRisk.CAUTION:
            return f"⚠️ CAUTION: Verify dosage for {medication} with healthcare provider."
        else:
            return f"Dosage for {medication} appears to be within typical range."
    
    def parse_dosage_from_text(self, text: str) -> List[Dict[str, any]]:
        """
        Parse dosage information from text.
        
        Args:
            text: Text containing dosage information
            
        Returns:
            List of parsed dosages
        """
        dosages = []
        
        # Pattern: medication dose unit frequency
        # Example: "metformin 500mg twice daily"
        pattern = r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?)\s+(.*?)(?:\.|,|$)'
        
        matches = re.findall(pattern, text.lower())
        
        for match in matches:
            medication, dose, unit, frequency = match
            dosage_info = {
                "medication": medication,
                "dose": float(dose),
                "unit": unit,
                "frequency": frequency.strip()
            }
            # Add aliases for test compatibility
            dosage_info["drug"] = dosage_info["medication"]
            dosage_info["amount"] = dosage_info["dose"]
            dosages.append(dosage_info)
        
        return dosages


class DosageRecommendationEngine:
    """
    Provide dosage recommendations based on guidelines.
    Note: This is educational only - not for actual prescribing.
    """
    
    STANDARD_DOSING = {
        "metformin": {
            "adult": "Start 500mg once or twice daily, titrate to 1000-2000mg daily",
            "max": "2550mg daily",
            "considerations": "Take with meals to reduce GI side effects"
        },
        "lisinopril": {
            "adult": "Start 10mg once daily, usual range 10-40mg daily",
            "max": "40mg daily",
            "considerations": "Monitor blood pressure and renal function"
        }
    }
    
    def __init__(self):
        """Initialize dosage recommendation engine."""
        logger.info("Initialized DosageRecommendationEngine")
    
    def get_recommendation(self, medication: str) -> Optional[Dict[str, str]]:
        """
        Get dosage recommendation for medication.
        
        Args:
            medication: Medication name
            
        Returns:
            Dosage recommendation dict or None
        """
        medication_lower = medication.lower()
        
        if medication_lower in self.STANDARD_DOSING:
            recommendation = self.STANDARD_DOSING[medication_lower].copy()
            recommendation["disclaimer"] = (
                "This is general information only. "
                "Always consult current prescribing information and "
                "consider patient-specific factors."
            )
            return recommendation
        
        return None
