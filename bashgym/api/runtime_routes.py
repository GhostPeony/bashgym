"""Runtime observation routes for work launched outside BashGym's API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request

from bashgym._compat import UTC
from bashgym.api.runtime_observer import RuntimeObserver

router = APIRouter(prefix="/api/runtime", tags=["Runtime"])


@router.get("/jobs")
def list_runtime_jobs(request: Request):
    observer = getattr(request.app.state, "runtime_observer", None)
    if observer is None:
        observer = RuntimeObserver(Path.cwd())
        request.app.state.runtime_observer = observer
    return {
        "jobs": observer.list_jobs(),
        "polled_at": datetime.now(UTC).isoformat(),
    }
