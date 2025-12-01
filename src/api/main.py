"""
Main FastAPI application for Medical AI Assistant.
Provides RESTful API endpoints with authentication, rate limiting, and monitoring.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from .models import HealthResponse, ErrorResponse
from .routes import ner_router, classification_router, rag_router, safety_router
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware

# Setup logger
logger = setup_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'medical_ai_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'medical_ai_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ERROR_COUNT = Counter(
    'medical_ai_errors_total',
    'Total number of errors',
    ['error_type']
)


# Application state management
class AppState:
    """Global application state."""
    
    def __init__(self):
        self.config = None
        self.ner_model = None
        self.classifier_model = None
        self.rag_system = None
        self.safety_guardrails = None
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize all models and systems."""
        logger.info("Initializing application state...")
        
        try:
            # Load configuration
            config_loader = ConfigLoader()
            self.config = {
                "api": config_loader.load_api_config(),
                "ner": config_loader.load_ner_config(),
                "classifier": config_loader.load_classifier_config(),
                "rag": config_loader.load_rag_config(),
                "llm": config_loader.load_llm_config(),
                "safety": config_loader.load_safety_config(),
                "mlops": config_loader.load_mlops_config(),
            }
            
            # Initialize models (lazy loading - will be loaded on first request)
            logger.info("Models will be loaded lazily on first request")
            
            logger.info("Application state initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize application state: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down application...")
        # Cleanup any resources here
        logger.info("Application shutdown complete")


# Global app state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    await app_state.initialize()
    yield
    # Shutdown
    await app_state.shutdown()


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Medical AI Assistant API",
        description="Production API for medical NER, classification, RAG, and safety guardrails",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middlewares
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # Add metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Collect Prometheus metrics for each request."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        method = request.method
        endpoint = request.url.path
        status_code = response.status_code
        
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        ERROR_COUNT.labels(error_type="validation_error").inc()
        logger.warning(f"Validation error: {exc.errors()}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Validation Error",
                message="Invalid request data",
                details=exc.errors()
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        ERROR_COUNT.labels(error_type="internal_error").inc()
        logger.error(f"Internal error: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                message="An unexpected error occurred",
                details=str(exc)
            ).dict()
        )
    
    # Include routers
    app.include_router(ner_router, prefix="/api/v1/ner", tags=["NER"])
    app.include_router(classification_router, prefix="/api/v1/classification", tags=["Classification"])
    app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
    app.include_router(safety_router, prefix="/api/v1/safety", tags=["Safety"])
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Medical AI Assistant API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint for monitoring.
        
        Returns:
            HealthResponse: System health status
        """
        uptime = time.time() - app_state.start_time
        
        return HealthResponse(
            status="healthy",
            uptime_seconds=uptime,
            services={
                "ner": "loaded" if app_state.ner_model else "not_loaded",
                "classification": "loaded" if app_state.classifier_model else "not_loaded",
                "rag": "loaded" if app_state.rag_system else "not_loaded",
                "safety": "loaded" if app_state.safety_guardrails else "not_loaded"
            }
        )
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """
        Prometheus metrics endpoint.
        
        Returns:
            Response: Prometheus metrics in text format
        """
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Readiness probe
    @app.get("/ready")
    async def readiness():
        """
        Readiness probe for Kubernetes.
        
        Returns:
            Dict: Readiness status
        """
        return {"status": "ready"}
    
    # Liveness probe
    @app.get("/live")
    async def liveness():
        """
        Liveness probe for Kubernetes.
        
        Returns:
            Dict: Liveness status
        """
        return {"status": "alive"}
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Load API config
    config_loader = ConfigLoader()
    api_config = config_loader.load_config("api_config")
    
    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", False),
        workers=api_config.get("workers", 1),
        log_level=api_config.get("log_level", "info")
    )
