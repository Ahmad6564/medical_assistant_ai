"""
Medical AI Assistant API module.
Provides RESTful API endpoints for all AI services.
"""

from .main import app, create_app
from .models import (
    HealthResponse,
    ErrorResponse,
    NERRequest,
    NERResponse,
    ClassificationRequest,
    ClassificationResponse,
    RAGRequest,
    RAGResponse,
    SafetyCheckRequest,
    SafetyCheckResponse
)

__all__ = [
    'app',
    'create_app',
    'HealthResponse',
    'ErrorResponse',
    'NERRequest',
    'NERResponse',
    'ClassificationRequest',
    'ClassificationResponse',
    'RAGRequest',
    'RAGResponse',
    'SafetyCheckRequest',
    'SafetyCheckResponse'
]
