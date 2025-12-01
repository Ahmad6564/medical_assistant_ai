"""
Pydantic models for API request/response schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


# Base models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    services: Dict[str, str] = Field(..., description="Status of each service")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Additional error details")


# NER models
class Entity(BaseModel):
    """Medical entity."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label/type")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    score: Optional[float] = Field(None, description="Confidence score", ge=0.0, le=1.0)


class NERRequest(BaseModel):
    """NER request."""
    text: str = Field(..., description="Input text for entity extraction", min_length=1, max_length=10000)
    model_type: Optional[str] = Field("transformer", description="Model type: 'transformer' or 'bilstm_crf'")
    include_linking: Optional[bool] = Field(False, description="Include entity linking to medical ontologies")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class NERResponse(BaseModel):
    """NER response."""
    entities: List[Entity] = Field(..., description="Extracted entities")
    text: str = Field(..., description="Original input text")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(..., description="Model used for extraction")


# Classification models
class ClassificationRequest(BaseModel):
    """Clinical text classification request."""
    text: str = Field(..., description="Clinical text to classify", min_length=1, max_length=10000)
    top_k: Optional[int] = Field(3, description="Number of top predictions to return", ge=1, le=10)
    threshold: Optional[float] = Field(0.5, description="Confidence threshold", ge=0.0, le=1.0)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class ClassificationPrediction(BaseModel):
    """Single classification prediction."""
    label: str = Field(..., description="Predicted label")
    score: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    probability: float = Field(..., description="Probability", ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    """Classification response."""
    predictions: List[ClassificationPrediction] = Field(..., description="Classification predictions")
    text: str = Field(..., description="Original input text")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(..., description="Model used for classification")


# RAG models
class RAGRequest(BaseModel):
    """RAG question answering request."""
    query: str = Field(..., description="User question or query", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    use_chain_of_thought: Optional[bool] = Field(False, description="Use chain-of-thought reasoning")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class RetrievedDocument(BaseModel):
    """Retrieved document."""
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class RAGResponse(BaseModel):
    """RAG response."""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    retrieved_documents: List[RetrievedDocument] = Field(..., description="Retrieved source documents")
    num_sources: int = Field(..., description="Number of sources used")
    reasoning_steps: Optional[List[str]] = Field(None, description="Chain-of-thought reasoning steps")
    disclaimers: List[str] = Field(default_factory=list, description="Safety disclaimers")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# Safety models
class SafetyLevel(str, Enum):
    """Safety level enumeration."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class SafetyCheckRequest(BaseModel):
    """Safety check request."""
    text: str = Field(..., description="Text to check for safety", min_length=1, max_length=10000)
    check_type: Optional[str] = Field("both", description="Check type: 'query', 'response', or 'both'")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class SafetyCheckResponse(BaseModel):
    """Safety check response."""
    text: str = Field(..., description="Original text")
    safety_level: SafetyLevel = Field(..., description="Overall safety level")
    is_safe: bool = Field(..., description="Whether text is safe")
    emergency_detected: bool = Field(..., description="Emergency situation detected")
    emergency_level: Optional[str] = Field(None, description="Emergency severity level")
    warnings: List[str] = Field(default_factory=list, description="Safety warnings")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")
    prohibited_claims: List[str] = Field(default_factory=list, description="Prohibited claims detected")
    required_disclaimers: List[str] = Field(default_factory=list, description="Required disclaimers")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# Document upload models
class DocumentUploadRequest(BaseModel):
    """Document upload request for RAG system."""
    content: str = Field(..., description="Document content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    source: Optional[str] = Field(None, description="Document source")


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    success: bool = Field(..., description="Upload success status")
    document_id: str = Field(..., description="Unique document identifier")
    num_chunks: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")


# Batch processing models
class BatchNERRequest(BaseModel):
    """Batch NER request."""
    texts: List[str] = Field(..., description="List of texts to process", min_items=1, max_items=100)
    model_type: Optional[str] = Field("transformer", description="Model type")


class BatchNERResponse(BaseModel):
    """Batch NER response."""
    results: List[NERResponse] = Field(..., description="List of NER results")
    total_processed: int = Field(..., description="Total texts processed")
    total_processing_time_ms: float = Field(..., description="Total processing time")


# Authentication models
class TokenRequest(BaseModel):
    """Token request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")


class APIKeyResponse(BaseModel):
    """API key response."""
    api_key: str = Field(..., description="Generated API key")
    name: str = Field(..., description="API key name")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
