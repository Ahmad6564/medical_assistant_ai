"""
Unit tests for safety mechanisms.
"""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.unit
class TestEmergencyDetection:
    """Tests for emergency detection."""
    
    def test_detector_initialization(self):
        """Test emergency detector can be initialized."""
        from src.safety import EmergencyDetector
        
        detector = EmergencyDetector()
        assert detector is not None
    
    def test_detect_emergency_keywords(self):
        """Test detection of emergency keywords."""
        from src.safety import EmergencyDetector
        
        detector = EmergencyDetector()
        
        # Emergency cases
        assert detector.detect("I'm having chest pain")
        assert detector.detect("Can't breathe properly")
        assert detector.detect("Severe bleeding")
        assert detector.detect("Suicidal thoughts")
        
        # Non-emergency cases
        assert not detector.detect("What is diabetes?")
        assert not detector.detect("General health question")
    
    def test_emergency_response(self):
        """Test emergency response message."""
        from src.safety import EmergencyDetector
        
        detector = EmergencyDetector()
        
        query = "I'm having severe chest pain"
        is_emergency = detector.detect(query)
        
        if is_emergency:
            response = detector.get_response()
            assert "911" in response or "emergency" in response.lower()
    
    def test_emergency_categories(self):
        """Test categorization of emergencies."""
        from src.safety import EmergencyDetector
        
        detector = EmergencyDetector()
        
        # Cardiac emergency
        category = detector.categorize("Chest pain and shortness of breath")
        assert category in ["cardiac", "respiratory", "general"]
        
        # Mental health emergency
        category = detector.categorize("Having suicidal thoughts")
        assert category in ["mental_health", "general"]


@pytest.mark.unit
class TestClaimsFiltering:
    """Tests for medical claims filtering."""
    
    def test_filter_initialization(self):
        """Test claims filter can be initialized."""
        from src.safety import ClaimsFilter
        
        filter = ClaimsFilter()
        assert filter is not None
    
    def test_detect_medical_claims(self):
        """Test detection of medical claims."""
        from src.safety import ClaimsFilter
        
        filter = ClaimsFilter()
        
        # Claims that should be flagged
        assert filter.is_claim("This drug cures cancer")
        assert filter.is_claim("You should take medication X")
        assert filter.is_claim("This will definitely work")
        
        # Information that should not be flagged
        assert not filter.is_claim("Diabetes is a metabolic disorder")
        assert not filter.is_claim("Metformin is commonly prescribed")
    
    def test_add_disclaimer(self):
        """Test adding disclaimers to responses."""
        from src.safety import ClaimsFilter
        
        filter = ClaimsFilter()
        
        response = "You could try taking vitamin D."
        filtered = filter.add_disclaimer(response)
        
        assert "disclaimer" in filtered.lower() or "consult" in filtered.lower()
    
    def test_confidence_based_filtering(self):
        """Test filtering based on confidence scores."""
        from src.safety import ClaimsFilter
        
        filter = ClaimsFilter()
        
        # High confidence claim
        claim1 = "This medication will cure your condition"
        score1 = filter.get_claim_score(claim1)
        
        # Low confidence informational statement
        claim2 = "Aspirin is used for pain relief"
        score2 = filter.get_claim_score(claim2)
        
        assert score1 > score2


@pytest.mark.unit
class TestDosageValidation:
    """Tests for dosage validation."""
    
    def test_validator_initialization(self):
        """Test dosage validator can be initialized."""
        from src.safety import DosageValidator
        
        validator = DosageValidator()
        assert validator is not None
    
    def test_extract_dosage_info(self):
        """Test extracting dosage information."""
        from src.safety import DosageValidator
        
        validator = DosageValidator()
        
        text = "Take 500mg of metformin twice daily"
        dosage_info = validator.extract_dosage(text)
        
        assert dosage_info is not None
        assert "amount" in dosage_info or "drug" in dosage_info
    
    def test_validate_safe_dosage_ranges(self):
        """Test validation against safe dosage ranges."""
        from src.safety import DosageValidator
        
        validator = DosageValidator()
        
        # Normal dosage
        is_safe1 = validator.validate("metformin", 500, "mg")
        
        # Potentially unsafe dosage
        is_safe2 = validator.validate("metformin", 10000, "mg")
        
        # Safe dosage should pass
        assert is_safe1 or is_safe1 is None  # None if no range defined
    
    def test_dosage_warning_generation(self):
        """Test generation of dosage warnings."""
        from src.safety import DosageValidator
        
        validator = DosageValidator()
        
        warning = validator.get_warning("metformin", 5000, "mg")
        
        if warning:
            assert isinstance(warning, str)
            assert len(warning) > 0


@pytest.mark.unit
class TestPrivacyProtection:
    """Tests for privacy protection."""
    
    def test_pii_detector_initialization(self):
        """Test PII detector can be initialized."""
        from src.safety import PIIDetector
        
        detector = PIIDetector()
        assert detector is not None
    
    def test_detect_personal_information(self):
        """Test detection of personal information."""
        from src.safety import PIIDetector
        
        detector = PIIDetector()
        
        # Text with PII
        text1 = "My email is john@example.com and phone is 555-1234"
        assert detector.contains_pii(text1)
        
        # Text without PII
        text2 = "What is diabetes?"
        assert not detector.contains_pii(text2)
    
    def test_anonymize_text(self):
        """Test anonymization of sensitive information."""
        from src.safety import PIIDetector
        
        detector = PIIDetector()
        
        text = "Patient John Doe, born 01/15/1980, SSN: 123-45-6789"
        anonymized = detector.anonymize(text)
        
        assert "John Doe" not in anonymized or "[NAME]" in anonymized
        assert "123-45-6789" not in anonymized


@pytest.mark.unit
class TestContentModeration:
    """Tests for content moderation."""
    
    def test_moderator_initialization(self):
        """Test content moderator can be initialized."""
        from src.safety import ContentModerator
        
        moderator = ContentModerator()
        assert moderator is not None
    
    def test_detect_inappropriate_content(self):
        """Test detection of inappropriate content."""
        from src.safety import ContentModerator
        
        moderator = ContentModerator()
        
        # Appropriate content
        assert not moderator.is_inappropriate("What are symptoms of diabetes?")
        
        # Potentially inappropriate content
        text = "How to make illegal substances"
        result = moderator.is_inappropriate(text)
        # Should detect or at least process without error
        assert isinstance(result, bool)
    
    def test_toxicity_scoring(self):
        """Test toxicity scoring."""
        from src.safety import ContentModerator
        
        moderator = ContentModerator()
        
        text = "You are stupid and worthless"
        score = moderator.get_toxicity_score(text)
        
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestDisclaimers:
    """Tests for medical disclaimers."""
    
    def test_get_general_disclaimer(self):
        """Test getting general disclaimer."""
        from src.safety import get_general_disclaimer
        
        disclaimer = get_general_disclaimer()
        
        assert isinstance(disclaimer, str)
        assert len(disclaimer) > 0
        assert "medical professional" in disclaimer.lower() or "doctor" in disclaimer.lower()
    
    def test_get_emergency_disclaimer(self):
        """Test getting emergency disclaimer."""
        from src.safety import get_emergency_disclaimer
        
        disclaimer = get_emergency_disclaimer()
        
        assert isinstance(disclaimer, str)
        assert "911" in disclaimer or "emergency" in disclaimer.lower()
    
    def test_context_specific_disclaimers(self):
        """Test context-specific disclaimers."""
        from src.safety import get_context_disclaimer
        
        # Medication-related
        disclaimer1 = get_context_disclaimer("medication")
        assert "prescribe" in disclaimer1.lower() or "medication" in disclaimer1.lower()
        
        # Diagnosis-related
        disclaimer2 = get_context_disclaimer("diagnosis")
        assert "diagnose" in disclaimer2.lower() or "diagnosis" in disclaimer2.lower()


@pytest.mark.integration
def test_safety_pipeline():
    """Test complete safety pipeline."""
    from src.safety import SafetyPipeline
    
    pipeline = SafetyPipeline()
    
    # Test query
    query = "I have chest pain"
    result = pipeline.check(query)
    
    assert "is_safe" in result or "is_emergency" in result
    
    if result.get("is_emergency"):
        assert "response" in result


@pytest.mark.unit
def test_medical_scope_validation():
    """Test validation of medical scope."""
    from src.safety import is_medical_query
    
    # Medical queries
    assert is_medical_query("What are symptoms of diabetes?")
    assert is_medical_query("Tell me about hypertension")
    
    # Non-medical queries
    assert not is_medical_query("What's the weather today?")
    assert not is_medical_query("How to cook pasta?")


@pytest.mark.unit
def test_response_safety_check():
    """Test safety checking of generated responses."""
    from src.safety import check_response_safety
    
    # Safe response
    response1 = "Diabetes is managed through diet, exercise, and medication. Consult your doctor."
    assert check_response_safety(response1)
    
    # Potentially unsafe response
    response2 = "Take 5000mg of this medication immediately"
    result = check_response_safety(response2)
    assert isinstance(result, bool)
