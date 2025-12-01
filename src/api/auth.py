"""
Authentication and authorization for API endpoints.
Implements JWT tokens and API key authentication.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from passlib.context import CryptContext

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthenticationError(Exception):
    """Authentication error."""
    pass


class User:
    """User model."""
    
    def __init__(self, username: str, email: str, is_active: bool = True, permissions: list = None):
        self.username = username
        self.email = email
        self.is_active = is_active
        self.permissions = permissions or []
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions or "admin" in self.permissions


class TokenManager:
    """JWT token management."""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token.
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            str: Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Dict: Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            if "expired" in str(e).lower():
                raise AuthenticationError("Token has expired")
            raise AuthenticationError(f"Invalid token: {e}")


class APIKeyManager:
    """API key management."""
    
    # In-memory storage (use database in production)
    _api_keys: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def create_api_key(cls, name: str, permissions: list = None, expires_in_days: Optional[int] = None) -> str:
        """
        Create a new API key.
        
        Args:
            name: API key name
            permissions: List of permissions
            expires_in_days: Expiration in days
            
        Returns:
            str: Generated API key
        """
        api_key = f"med_ai_{secrets.token_urlsafe(32)}"
        
        expiry = None
        if expires_in_days:
            expiry = datetime.utcnow() + timedelta(days=expires_in_days)
        
        cls._api_keys[api_key] = {
            "name": name,
            "permissions": permissions or [],
            "created_at": datetime.utcnow(),
            "expires_at": expiry,
            "is_active": True
        }
        
        logger.info(f"Created API key: {name}")
        return api_key
    
    @classmethod
    def verify_api_key(cls, api_key: str) -> Dict[str, Any]:
        """
        Verify API key.
        
        Args:
            api_key: API key to verify
            
        Returns:
            Dict: API key metadata
            
        Raises:
            AuthenticationError: If API key is invalid
        """
        if api_key not in cls._api_keys:
            raise AuthenticationError("Invalid API key")
        
        key_data = cls._api_keys[api_key]
        
        if not key_data["is_active"]:
            raise AuthenticationError("API key has been deactivated")
        
        if key_data["expires_at"] and datetime.utcnow() > key_data["expires_at"]:
            raise AuthenticationError("API key has expired")
        
        return key_data
    
    @classmethod
    def revoke_api_key(cls, api_key: str):
        """Revoke an API key."""
        if api_key in cls._api_keys:
            cls._api_keys[api_key]["is_active"] = False
            logger.info(f"Revoked API key: {cls._api_keys[api_key]['name']}")


class UserDatabase:
    """
    User database (in-memory for demo - use real database in production).
    """
    
    # Pre-hashed passwords (passwords stored as bcrypt hashes for security)
    # NOTE: Change these credentials in production!
    _users: Dict[str, Dict[str, Any]] = {
        "admin": {
            "username": "admin",
            "email": "admin@medical-ai.com",
            "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqVr/qvQK6",
            "is_active": True,
            "permissions": ["admin", "ner", "classification", "rag", "safety"]
        },
        "demo": {
            "username": "demo",
            "email": "demo@medical-ai.com",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
            "is_active": True,
            "permissions": ["ner", "classification", "rag"]
        }
    }
    
    @classmethod
    def get_user(cls, username: str) -> Optional[User]:
        """Get user by username."""
        user_data = cls._users.get(username)
        if not user_data:
            return None
        
        return User(
            username=user_data["username"],
            email=user_data["email"],
            is_active=user_data["is_active"],
            permissions=user_data["permissions"]
        )
    
    @classmethod
    def verify_password(cls, username: str, password: str) -> bool:
        """Verify user password."""
        user_data = cls._users.get(username)
        if not user_data:
            return False
        
        return pwd_context.verify(password, user_data["hashed_password"])
    
    @classmethod
    def authenticate_user(cls, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        if not cls.verify_password(username, password):
            return None
        
        return cls.get_user(username)


# Dependency functions
async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> User:
    """
    Get current user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        payload = TokenManager.verify_token(token)
        
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        user = UserDatabase.get_user(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
        
        return user
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[Dict[str, Any]]:
    """
    Get user from API key.
    
    Args:
        api_key: API key from header
        
    Returns:
        Dict: API key metadata or None
    """
    if not api_key:
        return None
    
    try:
        key_data = APIKeyManager.verify_api_key(api_key)
        return key_data
    except AuthenticationError:
        return None


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_data: Optional[Dict] = Depends(get_current_user_from_api_key)
) -> User:
    """
    Get current user from either JWT token or API key.
    
    Args:
        token_user: User from JWT token
        api_key_data: Data from API key
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try token authentication first
    if token_user:
        return token_user
    
    # Try API key authentication
    if api_key_data:
        # Create a user object from API key data
        return User(
            username=api_key_data["name"],
            email=f"{api_key_data['name']}@api-key",
            is_active=True,
            permissions=api_key_data["permissions"]
        )
    
    # No valid authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"}
    )


def require_permission(permission: str):
    """
    Dependency to require a specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def permission_checker(current_user: User = Depends(get_current_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required"
            )
        return current_user
    
    return permission_checker


# Optional authentication (for public endpoints with enhanced features for authenticated users)
async def get_optional_user(
    api_key_data: Optional[Dict] = Depends(get_current_user_from_api_key)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.
    Useful for endpoints that work for both authenticated and anonymous users.
    
    Args:
        api_key_data: Data from API key
        
    Returns:
        User or None
    """
    if api_key_data:
        return User(
            username=api_key_data["name"],
            email=f"{api_key_data['name']}@api-key",
            is_active=True,
            permissions=api_key_data["permissions"]
        )
    return None
