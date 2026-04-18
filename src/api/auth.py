"""
API Key authentication for the Procurement Digital Twin API.

Configuration
-------------
Set API_KEYS to a comma-separated list of valid keys:
  API_KEYS=key1,key2,key3

If API_KEYS is empty or unset, authentication is DISABLED (development mode).
A startup warning is logged when auth is disabled.

Usage
-----
Add as a FastAPI dependency on any route that should be protected:
  @app.post("/simulate", dependencies=[Depends(require_api_key)])
"""

import os
import logging
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_raw = os.environ.get("API_KEYS", "")
_API_KEYS: frozenset[str] = frozenset(k.strip() for k in _raw.split(",") if k.strip())
AUTH_ENABLED: bool = bool(_API_KEYS)

if not AUTH_ENABLED:
    logger.warning(
        "API_KEYS is not set — authentication is DISABLED. "
        "Set API_KEYS=<key1,key2,...> to enable."
    )
else:
    logger.info("API key authentication enabled (%d key(s) configured).", len(_API_KEYS))


async def require_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency that enforces API key authentication.

    When AUTH_ENABLED is False (no API_KEYS configured) every request passes
    through without inspection — useful for local development.

    Returns the validated key string so route handlers can log the caller.
    """
    if not AUTH_ENABLED:
        return ""
    if not x_api_key or x_api_key not in _API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass a valid key in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key
