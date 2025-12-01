"""
Query processing and enhancement for better retrieval.
Includes query expansion, rewriting, and medical term normalization.
"""

import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedQuery:
    """Processed query with enhancements."""
    original: str
    cleaned: str
    expanded: List[str]
    keywords: List[str]
    medical_terms: List[str]


class QueryProcessor:
    """Process and enhance queries for better retrieval."""
    
    # Common medical abbreviations and their expansions
    MEDICAL_ABBREVIATIONS = {
        "htn": "hypertension",
        "dm": "diabetes mellitus",
        "mi": "myocardial infarction",
        "cad": "coronary artery disease",
        "copd": "chronic obstructive pulmonary disease",
        "chf": "congestive heart failure",
        "afib": "atrial fibrillation",
        "cvd": "cardiovascular disease",
        "ckd": "chronic kidney disease",
        "gerd": "gastroesophageal reflux disease",
        "ra": "rheumatoid arthritis",
        "oa": "osteoarthritis",
        "ibd": "inflammatory bowel disease",
        "tb": "tuberculosis",
        "hiv": "human immunodeficiency virus",
        "copd": "chronic obstructive pulmonary disease",
        "ca": "cancer",
        "cva": "cerebrovascular accident",
        "dvt": "deep vein thrombosis",
        "pe": "pulmonary embolism",
        "uri": "upper respiratory infection",
        "uti": "urinary tract infection",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "n/v": "nausea and vomiting",
        "abd": "abdominal",
        "rx": "prescription",
        "tx": "treatment",
        "dx": "diagnosis",
        "hx": "history",
        "sx": "symptoms",
        "pt": "patient"
    }
    
    # Medical synonyms for query expansion
    MEDICAL_SYNONYMS = {
        "heart attack": ["myocardial infarction", "mi", "cardiac arrest"],
        "high blood pressure": ["hypertension", "htn", "elevated bp"],
        "diabetes": ["diabetes mellitus", "dm", "high blood sugar"],
        "stroke": ["cerebrovascular accident", "cva", "brain attack"],
        "chest pain": ["angina", "cp", "thoracic pain"],
        "shortness of breath": ["dyspnea", "sob", "breathlessness"],
        "heart failure": ["congestive heart failure", "chf", "cardiac failure"],
        "kidney disease": ["renal disease", "ckd", "nephropathy"],
        "liver disease": ["hepatic disease", "cirrhosis"],
        "lung disease": ["pulmonary disease", "respiratory disease"],
        "cancer": ["malignancy", "carcinoma", "neoplasm", "tumor"],
        "infection": ["sepsis", "bacterial infection", "viral infection"],
        "pain": ["discomfort", "ache", "soreness"],
        "medication": ["drug", "medicine", "prescription", "therapy"],
        "treatment": ["therapy", "intervention", "management"],
        "test": ["examination", "lab", "diagnostic test"],
        "surgery": ["operation", "surgical procedure", "intervention"]
    }
    
    # Stop words specific to medical queries
    MEDICAL_STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "how", "why", "when",
        "where", "can", "does", "do", "will", "would", "should",
        "about", "for", "of", "in", "on", "at", "to", "with"
    }
    
    def __init__(
        self,
        expand_abbreviations: bool = True,
        expand_synonyms: bool = True,
        max_expansions: int = 3
    ):
        """
        Initialize query processor.
        
        Args:
            expand_abbreviations: Whether to expand medical abbreviations
            expand_synonyms: Whether to expand with synonyms
            max_expansions: Maximum number of query expansions
        """
        self.expand_abbreviations = expand_abbreviations
        self.expand_synonyms = expand_synonyms
        self.max_expansions = max_expansions
        
        logger.info("Initialized QueryProcessor")
    
    def process(self, query: str) -> ProcessedQuery:
        """
        Process and enhance query.
        
        Args:
            query: Raw query string
            
        Returns:
            ProcessedQuery object
        """
        # Clean query
        cleaned = self._clean_query(query)
        
        # Extract medical terms
        medical_terms = self._extract_medical_terms(cleaned)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned)
        
        # Expand query
        expanded = self._expand_query(cleaned)
        
        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            expanded=expanded[:self.max_expansions],
            keywords=keywords,
            medical_terms=medical_terms
        )
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize query.
        
        Args:
            query: Raw query
            
        Returns:
            Cleaned query
        """
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters (keep alphanumeric and common punctuation)
        cleaned = re.sub(r'[^\w\s\-\/]', '', cleaned)
        
        return cleaned
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """
        Extract medical terms from query.
        
        Args:
            query: Cleaned query
            
        Returns:
            List of medical terms
        """
        terms = []
        words = query.split()
        
        # Check for abbreviations
        for word in words:
            if word in self.MEDICAL_ABBREVIATIONS:
                terms.append(word)
                if self.expand_abbreviations:
                    terms.append(self.MEDICAL_ABBREVIATIONS[word])
        
        # Check for multi-word medical terms
        for term in self.MEDICAL_SYNONYMS.keys():
            if term in query:
                terms.append(term)
        
        return list(set(terms))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query (remove stop words).
        
        Args:
            query: Cleaned query
            
        Returns:
            List of keywords
        """
        words = query.split()
        
        # Filter out stop words
        keywords = [
            word for word in words
            if word not in self.MEDICAL_STOP_WORDS
            and len(word) > 2
        ]
        
        return keywords
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Cleaned query
            
        Returns:
            List of expanded queries
        """
        expanded = [query]
        
        if not self.expand_synonyms:
            return expanded
        
        # Expand abbreviations
        if self.expand_abbreviations:
            expanded_abbrev = query
            for abbrev, expansion in self.MEDICAL_ABBREVIATIONS.items():
                if abbrev in query.split():
                    expanded_abbrev = expanded_abbrev.replace(abbrev, expansion)
            
            if expanded_abbrev != query:
                expanded.append(expanded_abbrev)
        
        # Expand with synonyms
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in query:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                    expanded_query = query.replace(term, synonym)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)
                        
                        if len(expanded) >= self.max_expansions:
                            return expanded
        
        return expanded


class QueryRewriter:
    """
    Rewrite queries to improve retrieval quality.
    Handles complex medical queries and question formats.
    """
    
    QUESTION_PATTERNS = [
        (r"what (?:is|are) (?:the )?(.+)", r"\1"),
        (r"how (?:to|do i) (.+)", r"\1"),
        (r"why (?:do|does) (.+)", r"\1 cause reason"),
        (r"when (?:should|to) (.+)", r"\1 timing indication"),
        (r"where (?:is|are) (.+)", r"\1 location"),
        (r"can (?:i|you) (.+)", r"\1 possibility"),
        (r"what (?:are|is) (?:the )?(?:symptoms|signs) of (.+)", r"\1 symptoms manifestations"),
        (r"how (?:to )?treat (.+)", r"\1 treatment management therapy"),
        (r"what (?:causes|leads to) (.+)", r"\1 etiology pathogenesis"),
        (r"how (?:to )?diagnose (.+)", r"\1 diagnosis diagnostic criteria"),
    ]
    
    def __init__(self):
        """Initialize query rewriter."""
        logger.info("Initialized QueryRewriter")
    
    def rewrite(self, query: str) -> str:
        """
        Rewrite query to declarative form.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        query_lower = query.lower().strip()
        
        # Try to match question patterns
        for pattern, replacement in self.QUESTION_PATTERNS:
            match = re.match(pattern, query_lower)
            if match:
                rewritten = re.sub(pattern, replacement, query_lower)
                logger.debug(f"Rewrote query: '{query}' -> '{rewritten}'")
                return rewritten
        
        # If no pattern matches, return original
        return query
    
    def generate_variations(self, query: str) -> List[str]:
        """
        Generate query variations for better coverage.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add rewritten version
        rewritten = self.rewrite(query)
        if rewritten != query:
            variations.append(rewritten)
        
        # Add question form if declarative
        if not any(query.lower().startswith(q) for q in ["what", "how", "why", "when", "where"]):
            variations.append(f"what is {query}")
            variations.append(f"how to treat {query}")
        
        return variations


class MedicalQueryEnhancer:
    """
    Complete query enhancement pipeline.
    Combines processing, rewriting, and expansion.
    """
    
    def __init__(
        self,
        enable_processing: bool = True,
        enable_rewriting: bool = True,
        enable_expansion: bool = True
    ):
        """
        Initialize query enhancer.
        
        Args:
            enable_processing: Enable query processing
            enable_rewriting: Enable query rewriting
            enable_expansion: Enable query expansion
        """
        self.enable_processing = enable_processing
        self.enable_rewriting = enable_rewriting
        self.enable_expansion = enable_expansion
        
        if enable_processing:
            self.processor = QueryProcessor()
        
        if enable_rewriting:
            self.rewriter = QueryRewriter()
        
        logger.info("Initialized MedicalQueryEnhancer")
    
    def enhance(self, query: str) -> Dict[str, any]:
        """
        Enhance query using all available methods.
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with enhanced query information
        """
        result = {
            "original": query,
            "processed": None,
            "rewritten": None,
            "variations": [],
            "keywords": [],
            "medical_terms": []
        }
        
        # Process query
        if self.enable_processing:
            processed = self.processor.process(query)
            result["processed"] = processed.cleaned
            result["keywords"] = processed.keywords
            result["medical_terms"] = processed.medical_terms
            result["variations"].extend(processed.expanded)
        
        # Rewrite query
        if self.enable_rewriting:
            rewritten = self.rewriter.rewrite(query)
            result["rewritten"] = rewritten
            if rewritten not in result["variations"]:
                result["variations"].append(rewritten)
        
        # Remove duplicates
        result["variations"] = list(dict.fromkeys(result["variations"]))
        
        return result
    
    def get_best_query(self, query: str) -> str:
        """
        Get the best enhanced version of the query.
        
        Args:
            query: Original query
            
        Returns:
            Best enhanced query
        """
        enhanced = self.enhance(query)
        
        # Prefer rewritten over processed over original
        if enhanced["rewritten"] and enhanced["rewritten"] != query:
            return enhanced["rewritten"]
        elif enhanced["processed"]:
            return enhanced["processed"]
        else:
            return query
