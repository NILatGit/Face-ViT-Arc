import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    if os.environ.get("DEV_MODE", "false").lower() == "true":
        return
    expected = os.environ.get("API_KEY", "")
    if not key or key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
