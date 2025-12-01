"""
Medical AI Assistant
A production-ready healthcare AI system for clinical decision support.
"""

__version__ = "1.0.0"
__author__ = "Muhammad Ahmad"

# Import core modules
from . import utils
from . import models
from . import rag
from . import llm
from . import safety

__all__ = [
    "utils",
    "models",
    "rag",
    "llm",
    "safety",
    "__version__",
    "__author__"
]