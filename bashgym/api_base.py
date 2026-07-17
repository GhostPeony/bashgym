"""Canonical BashGym API base URL validation."""

from __future__ import annotations

import urllib.parse
import urllib.request
from ipaddress import ip_address
from typing import Any


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep requests on the already-validated API authority and path."""

    def redirect_request(
        self,
        _req: urllib.request.Request,
        _fp: Any,
        _code: int,
        _msg: str,
        _headers: Any,
        _newurl: str,
    ) -> None:
        return None


def normalize_api_base(raw: str) -> str:
    """Return a credential-free HTTP(S) origin with the canonical ``/api`` path."""

    try:
        parsed = urllib.parse.urlsplit(str(raw).strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError("BashGym API base URL is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("BashGym API base must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("BashGym API base cannot contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("BashGym API base cannot contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path not in {"", "/api"}:
        raise ValueError("BashGym API base path must be /api")
    hostname = parsed.hostname
    is_loopback = hostname.casefold().rstrip(".") == "localhost"
    if not is_loopback:
        try:
            is_loopback = ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError("Plain HTTP BashGym API bases must use a loopback host")
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urllib.parse.urlunsplit((parsed.scheme, netloc, "/api", "", ""))


def open_api_url(
    request: str | urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    """Open one API request without following authority- or path-changing redirects."""

    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)
