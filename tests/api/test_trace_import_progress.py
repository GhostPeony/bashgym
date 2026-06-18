"""Trace import broadcasts per-item progress over pipeline:import.

Covers:
  - broadcast_import_progress helper sends a PIPELINE_IMPORT WSMessage
  - POST /api/traces/upload/import (chatgpt) broadcasts once per conversation
  - POST /api/traces/upload/import (mcp) broadcasts a single 1/1 progress event
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. broadcast_import_progress helper
# ---------------------------------------------------------------------------


async def test_broadcast_import_progress_sends_pipeline_import():
    from bashgym.api import websocket as ws

    with patch.object(ws.manager, "broadcast", new_callable=AsyncMock) as mock_broadcast:
        await ws.broadcast_import_progress(processed=2, total=10, current_item="session-2.json")

    mock_broadcast.assert_awaited_once()
    message = mock_broadcast.await_args.args[0]
    assert message.type == ws.MessageType.PIPELINE_IMPORT
    assert message.type.value == "pipeline:import"
    assert message.payload["processed"] == 2
    assert message.payload["total"] == 10
    assert message.payload["current_item"] == "session-2.json"
    assert message.payload["phase"] == "importing"


async def test_broadcast_import_progress_default_current_item():
    from bashgym.api import websocket as ws

    with patch.object(ws.manager, "broadcast", new_callable=AsyncMock) as mock_broadcast:
        await ws.broadcast_import_progress(processed=5, total=5)

    message = mock_broadcast.await_args.args[0]
    assert message.payload["processed"] == 5
    assert message.payload["total"] == 5
    assert message.payload["current_item"] == ""


# ---------------------------------------------------------------------------
# 2. Upload/import endpoint broadcasts per item
# ---------------------------------------------------------------------------

CHATGPT_IMPORTER_PATCH = "bashgym.trace_capture.importers.chatgpt.ChatGPTImporter"
MCP_IMPORTER_PATCH = "bashgym.trace_capture.importers.mcp_logs.MCPLogImporter"
PROGRESS_PATCH = "bashgym.api.routes.broadcast_import_progress"
TRACE_EVENT_PATCH = "bashgym.api.routes.broadcast_trace_event"


def _fake_chatgpt_result(i: int):
    from bashgym.trace_capture.importers.chatgpt import ChatGPTImportResult

    return ChatGPTImportResult(session_id=f"convo_{i}", title=f"Convo {i}", steps_imported=2)


class TestUploadImportProgress:
    def test_chatgpt_import_broadcasts_per_conversation(self, client):
        conversations = [
            {"id": f"c{i}", "title": f"Convo {i}", "mapping": {}, "current_node": ""}
            for i in range(3)
        ]
        payload = json.dumps(conversations).encode()

        mock_importer_cls = MagicMock()
        mock_importer_cls.return_value.import_conversation.side_effect = [
            _fake_chatgpt_result(i) for i in range(3)
        ]

        with (
            patch(CHATGPT_IMPORTER_PATCH, mock_importer_cls),
            patch(PROGRESS_PATCH, new_callable=AsyncMock) as mock_progress,
            patch(TRACE_EVENT_PATCH, new_callable=AsyncMock),
        ):
            resp = client.post(
                "/api/traces/upload/import",
                files={"file": ("conversations.json", payload, "application/json")},
                data={"source": "chatgpt"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["imported_count"] == 3
        assert data["failed_count"] == 0

        # One progress broadcast per conversation, 1-based processed counts
        assert mock_progress.await_count == 3
        for i, call in enumerate(mock_progress.await_args_list, start=1):
            assert call.kwargs["processed"] == i
            assert call.kwargs["total"] == 3
            assert call.kwargs["current_item"]

    def test_chatgpt_import_single_conversation_object(self, client):
        """A non-list conversations.json (single object) is one item: 1/1."""
        payload = json.dumps({"id": "c0", "title": "Solo", "mapping": {}}).encode()

        mock_importer_cls = MagicMock()
        mock_importer_cls.return_value.import_conversation.return_value = _fake_chatgpt_result(0)

        with (
            patch(CHATGPT_IMPORTER_PATCH, mock_importer_cls),
            patch(PROGRESS_PATCH, new_callable=AsyncMock) as mock_progress,
            patch(TRACE_EVENT_PATCH, new_callable=AsyncMock),
        ):
            resp = client.post(
                "/api/traces/upload/import",
                files={"file": ("conversations.json", payload, "application/json")},
                data={"source": "chatgpt"},
            )

        assert resp.status_code == 200
        assert mock_progress.await_count == 1
        call = mock_progress.await_args
        assert call.kwargs["processed"] == 1
        assert call.kwargs["total"] == 1

    def test_mcp_import_broadcasts_single_progress(self, client):
        mock_importer_cls = MagicMock()
        mock_importer_cls.return_value.import_from_file.return_value = MagicMock(
            error=None, skipped=False, steps_imported=4
        )

        with (
            patch(MCP_IMPORTER_PATCH, mock_importer_cls),
            patch(PROGRESS_PATCH, new_callable=AsyncMock) as mock_progress,
            patch(TRACE_EVENT_PATCH, new_callable=AsyncMock),
        ):
            resp = client.post(
                "/api/traces/upload/import",
                files={"file": ("mcp_log.json", b"{}", "application/json")},
                data={"source": "mcp"},
            )

        assert resp.status_code == 200
        assert resp.json()["imported_count"] == 1
        assert mock_progress.await_count == 1
        call = mock_progress.await_args
        assert call.kwargs["processed"] == 1
        assert call.kwargs["total"] == 1
