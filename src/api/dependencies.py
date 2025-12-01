from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

from ..utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)  # Changed to auto_error=False

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    Get current user from JWT token.
    Returns None if authentication is disabled or no credentials provided.
    """
    # Load config to check if auth is required
    config_loader = ConfigLoader()
    api_config = config_loader.load_api_config()
    
    # If authentication is not required, return a default user
    auth_required = api_config.get("authentication", {}).get("required", False) or \
                   api_config.get("enable_auth", False)
    
    if not auth_required:
        logger.debug("Authentication disabled - allowing request")
        return {"username": "anonymous", "permissions": ["all"]}
    
    # If auth is required but no credentials provided
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )
    
    # Validate token (your existing token validation logic)
    try:
        token = credentials.credentials
        # Add your token validation logic here
        return {"username": "user", "permissions": ["all"]}
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication credentials"
        )


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Optional authentication - never raises an error."""
    config_loader = ConfigLoader()
    api_config = config_loader.load_api_config()
    
    auth_required = api_config.get("authentication", {}).get("required", False) or \
                   api_config.get("enable_auth", False)
    
    if not auth_required or credentials is None:
        return {"username": "anonymous", "permissions": ["all"]}
    
    try:
        return await get_current_user(credentials)
    except:
        return {"username": "anonymous", "permissions": ["all"]}