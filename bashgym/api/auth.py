"""API key authentication middleware."""
import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/api/health",
    "/api/docs",
    "/api/redoc",
    "/api/openapi.json",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that checks X-API-Key header against BASHGYM_API_KEY env var.

    If BASHGYM_API_KEY is not set, authentication is disabled (dev mode).
    """

    async def dispatch(self, request: Request, call_next):
        api_key = os.environ.get("BASHGYM_API_KEY", "")

        # Skip auth if no key configured (dev mode) or WebSocket
        if not api_key or request.url.path.startswith("/ws"):
            return await call_next(request)

        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check API key
        request_key = request.headers.get("X-API-Key", "")
        if request_key != api_key:
            raise HTTPException(status_code=403, detail="Invalid or missing API key")

        return await call_next(request)
