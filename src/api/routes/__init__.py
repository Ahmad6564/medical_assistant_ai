"""
API routes package.
"""

from .ner import router as ner_router
from .classification import router as classification_router
from .rag import router as rag_router
from .safety import router as safety_router

__all__ = [
    'ner_router',
    'classification_router',
    'rag_router',
    'safety_router'
]
