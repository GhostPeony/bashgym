from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from bashgym.api import models_routes


@pytest.mark.asyncio
async def test_unimplemented_model_evaluation_never_claims_queued(monkeypatch):
    monkeypatch.setattr(
        models_routes, "get_registry", lambda: SimpleNamespace(get=lambda _: object())
    )
    with pytest.raises(HTTPException) as error:
        await models_routes.trigger_evaluation("registered-model")
    assert error.value.status_code == 501
    assert "No evaluation was queued" in error.value.detail


@pytest.mark.asyncio
async def test_unknown_model_evaluation_remains_not_found(monkeypatch):
    monkeypatch.setattr(models_routes, "get_registry", lambda: SimpleNamespace(get=lambda _: None))
    with pytest.raises(HTTPException) as error:
        await models_routes.trigger_evaluation("missing-model")
    assert error.value.status_code == 404
