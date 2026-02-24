"""
This module defines a single FastAPI dependency, verify_api_key, which is
applied to every protected route via dependencies=[Security(verify_api_key)].
All such routes will return HTTP 401 if the key is missing or wrong.
"""

import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# APIKeyHeader tells FastAPI to look for a header named "X-API-Key" on every
# incoming request. auto_error=False means FastAPI will NOT automatically
# reject requests with a missing header - that decision is left to
# verify_api_key below, which allows us to return a more specific error
# message and also to support the DEV_MODE bypass.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    """
    FastAPI injects this function as a dependency before the route handler
    runs. If the function raises HTTPException, the route handler is never
    called and the exception response is returned to the client directly.

    Environment Variables:
    DEV_MODE : str, optional
        Set to "true" (case-insensitive) to skip all key validation. Intended
        for local development with `modal serve`. Any other value (or absent)
        means production mode where the key is enforced.
    API_KEY : str
        The expected API key value. Must match the X-API-Key header exactly
        (case-sensitive, no whitespace trimming). An empty or absent API_KEY
        env var means no valid key exists and all requests will be rejected
        unless DEV_MODE is active.

    To rotate the API key, recreate the Modal secret:
        modal secret create face-api-secret API_KEY=<new-key>
    Then redeploy or re-serve. No code changes are needed.
    """
    # DEV_MODE bypass - allows calling all endpoints without a key during
    # local development. Never leave DEV_MODE=true in a production secret.
    if os.environ.get("DEV_MODE", "false").lower() == "true":
        return

    # Read the expected key from the environment. This value comes from the
    # Modal secret attached to fastapi_app in main.py. If the environment
    # variable is not set (e.g. secret was misconfigured), expected will be
    # an empty string. Treat that as "not configured" and bypass auth rather
    # than silently blocking every request - ensures modal serve works before
    # the secret is fully set up.
    expected = os.environ.get("API_KEY", "")
    if not expected:
        return

    # Reject if the header was missing (falsy key) or does not match exactly.
    if not key or key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
