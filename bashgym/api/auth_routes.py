"""GitHub OAuth authentication routes.

Flow: /api/auth/github → GitHub → /api/auth/github/callback → set cookie → redirect /
"""

import hashlib
import logging
import os
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from bashgym.api.database import (
    SESSION_MAX_AGE_DAYS,
    create_session,
    delete_session,
    get_session_user,
    upsert_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

COOKIE_NAME = "bashgym_session"
# In-memory store for OAuth state tokens (short-lived, CSRF protection)
_oauth_states: dict[str, float] = {}
_MAX_PENDING_STATES = 100  # cap to prevent memory DoS from spamming /api/auth/github


def _get_client_id() -> str:
    val = os.environ.get("GITHUB_CLIENT_ID", "")
    if not val:
        raise RuntimeError("GITHUB_CLIENT_ID not set")
    return val


def _get_client_secret() -> str:
    val = os.environ.get("GITHUB_CLIENT_SECRET", "")
    if not val:
        raise RuntimeError("GITHUB_CLIENT_SECRET not set")
    return val


def _set_session_cookie(response: Response, token: str, request: Request) -> None:
    """Set the httpOnly session cookie with appropriate security flags."""
    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=is_https,
        samesite="lax",
        path="/",
        max_age=SESSION_MAX_AGE_DAYS * 86400,
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/")


def _prune_stale_states() -> None:
    """Remove OAuth state tokens older than 10 minutes, and enforce max size."""
    import time
    cutoff = time.time() - 600
    stale = [k for k, v in _oauth_states.items() if v < cutoff]
    for k in stale:
        _oauth_states.pop(k, None)
    # If still over cap, drop oldest entries
    while len(_oauth_states) > _MAX_PENDING_STATES:
        oldest_key = min(_oauth_states, key=_oauth_states.get)  # type: ignore
        _oauth_states.pop(oldest_key, None)


@router.get("/github")
async def github_login(request: Request):
    """Redirect to GitHub OAuth authorization page."""
    import time

    _prune_stale_states()

    # Generate CSRF state token
    state = secrets.token_urlsafe(32)
    _oauth_states[hashlib.sha256(state.encode()).hexdigest()] = time.time()

    params = {
        "client_id": _get_client_id(),
        "scope": "read:user user:email",
        "state": state,
    }
    return RedirectResponse(
        url=f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}",
        status_code=302,
    )


@router.get("/github/callback")
async def github_callback(request: Request, code: str = "", state: str = ""):
    """Exchange GitHub OAuth code for access token, create session."""
    import time

    if not code or not state:
        return JSONResponse({"error": "Missing code or state"}, status_code=400)

    # Verify CSRF state
    state_hash = hashlib.sha256(state.encode()).hexdigest()
    if state_hash not in _oauth_states:
        return JSONResponse({"error": "Invalid or expired state parameter"}, status_code=403)
    _oauth_states.pop(state_hash, None)

    # Exchange code for access token
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": _get_client_id(),
                "client_secret": _get_client_secret(),
                "code": code,
            },
            headers={"Accept": "application/json"},
        )

        if token_resp.status_code != 200:
            logger.error(f"GitHub token exchange failed: {token_resp.status_code}")
            return JSONResponse({"error": "GitHub token exchange failed"}, status_code=502)

        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            error = token_data.get("error_description", token_data.get("error", "unknown"))
            logger.error(f"GitHub OAuth error: {error}")
            return JSONResponse({"error": f"GitHub OAuth error: {error}"}, status_code=400)

        # Fetch user profile
        auth_headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        user_resp = await client.get(GITHUB_USER_URL, headers=auth_headers)
        if user_resp.status_code != 200:
            return JSONResponse({"error": "Failed to fetch GitHub profile"}, status_code=502)

        gh_user = user_resp.json()

        # Fetch primary email if not public
        email = gh_user.get("email")
        if not email:
            emails_resp = await client.get(GITHUB_EMAILS_URL, headers=auth_headers)
            if emails_resp.status_code == 200:
                for e in emails_resp.json():
                    if e.get("primary") and e.get("verified"):
                        email = e["email"]
                        break

    # Validate required fields from GitHub
    github_id = gh_user.get("id")
    username = gh_user.get("login")
    if not github_id or not username:
        return JSONResponse({"error": "Invalid GitHub profile data"}, status_code=502)

    # Upsert user and create session
    user_id = upsert_user(
        github_id=github_id,
        username=username,
        display_name=gh_user.get("name"),
        avatar_url=gh_user.get("avatar_url"),
        email=email,
    )

    user_agent = request.headers.get("user-agent", "")
    raw_token = create_session(user_id, user_agent=user_agent)

    # Redirect to app root with session cookie
    response = RedirectResponse(url="/", status_code=302)
    _set_session_cookie(response, raw_token, request)
    return response


@router.get("/me")
async def get_current_user(request: Request):
    """Return the currently authenticated user, or 401."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    user = get_session_user(token)
    if not user:
        return JSONResponse({"error": "Session expired"}, status_code=401)

    return {
        "id": user["id"],
        "github_id": user["github_id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "avatar_url": user["avatar_url"],
        "email": user["email"],
    }


@router.post("/logout")
async def logout(request: Request):
    """Delete session and clear cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        delete_session(token)

    response = JSONResponse({"ok": True})
    _clear_session_cookie(response)
    return response
