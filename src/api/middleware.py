"""
Custom middleware for API.
Includes rate limiting, request logging, and security headers.
"""

import time
import logging
from typing import Dict, Callable
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": requests_per_minute,
            "last_update": datetime.utcnow()
        })
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get API key or use IP address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Use IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _refill_bucket(self, bucket: Dict):
        """Refill tokens based on time passed."""
        now = datetime.utcnow()
        time_passed = (now - bucket["last_update"]).total_seconds()
        
        # Refill rate: requests_per_minute / 60 = tokens per second
        tokens_to_add = (time_passed / 60) * self.requests_per_minute
        
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/live", "/metrics"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        bucket = self.buckets[client_id]
        
        # Refill tokens
        self._refill_bucket(bucket)
        
        # Check if request can proceed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(int(bucket["tokens"]))
            
            return response
        else:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate Limit Exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        # Generate request ID
        request_id = f"{int(time.time() * 1000)}-{id(request)}"
        
        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2)
                }
            )
            
            # Add request ID to response
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration * 1000, 2)
                },
                exc_info=True
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware with more control.
    """
    
    def __init__(self, app, allow_origins: list = None, allow_methods: list = None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS."""
        # Handle preflight requests
        if request.method == "OPTIONS":
            return Response(
                status_code=status.HTTP_204_NO_CONTENT,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                    "Access-Control-Max-Age": "3600"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        
        return response
