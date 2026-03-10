"""Authentication middleware supporting desktop (no-auth) and web (cookie/API key) modes.

Modes:
- Desktop / dev (BASHGYM_MODE != 'web'): all requests pass through, no auth required
- Web (BASHGYM_MODE == 'web'): API routes require a valid session cookie or X-API-Key header
"""

import hmac
import os

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from bashgym.api.auth_routes import COOKIE_NAME
from bashgym.api.database import get_session_user

# Paths that never require authentication
PUBLIC_PATHS = {
    "/api/health",
}

# Auth flow paths (must be accessible pre-login)
AUTH_PATH_PREFIX = "/api/auth/"

# Paths that require auth in web mode (even though they're not under /api/)
PROTECTED_NON_API_PATHS = {
    "/docs",
    "/redoc",
    "/openapi.json",
}


def _is_web_mode() -> bool:
    return os.environ.get("BASHGYM_MODE", "").lower() == "web"


def _requires_auth(path: str) -> bool:
    """Determine if a path requires authentication in web mode."""
    # Always public
    if path in PUBLIC_PATHS:
        return False

    # Auth flow endpoints (login, callback, etc.)
    if path.startswith(AUTH_PATH_PREFIX):
        return False

    # API endpoints require auth
    if path.startswith("/api/"):
        return True

    # Docs aliases that should be locked down
    if path in PROTECTED_NON_API_PATHS:
        return True

    # Everything else (SPA static files, assets) — no auth
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Dual-mode auth middleware.

    Desktop/dev: pass-through (no auth enforced).
    Web: session cookie or API key required for /api/* routes.
    """

    async def dispatch(self, request: Request, call_next):
        # --- Desktop / dev mode: no auth at all ---
        if not _is_web_mode():
            return await call_next(request)

        # --- Web mode ---
        path = request.url.path

        # WebSocket — auth checked separately during handshake
        if path.startswith("/ws"):
            return await call_next(request)

        # Check if this path requires auth
        if not _requires_auth(path):
            return await call_next(request)

        # CSRF protection for state-changing requests
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            if request.headers.get("X-Requested-With") != "XMLHttpRequest":
                # Allow multipart form uploads (they set their own content type)
                content_type = request.headers.get("content-type", "")
                if "multipart/form-data" not in content_type:
                    return JSONResponse(
                        {"detail": "Missing CSRF header"},
                        status_code=403,
                    )

        # --- Authenticate ---

        # 1. Try X-API-Key header (programmatic access)
        api_key = os.environ.get("BASHGYM_API_KEY", "")
        if api_key:
            request_key = request.headers.get("X-API-Key", "")
            if request_key and hmac.compare_digest(request_key, api_key):
                return await call_next(request)

        # 2. Try session cookie
        token = request.cookies.get(COOKIE_NAME)
        if token:
            user = get_session_user(token)
            if user:
                request.state.user = user
                return await call_next(request)

        # 3. Reject
        return JSONResponse(
            {"detail": "Authentication required"},
            status_code=401,
        )


# Keep old name as alias for backwards compat during transition
APIKeyMiddleware = AuthMiddleware
