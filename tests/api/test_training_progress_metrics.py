import asyncio
import json

from bashgym.api import websocket as ws


def _capture_broadcasts(monkeypatch):
    captured = []

    async def fake_broadcast(message):
        captured.append(message)

    monkeypatch.setattr(ws.manager, "broadcast", fake_broadcast)
    return captured


def test_on_progress_forwards_eval_and_session_distillation_metrics(tmp_path, monkeypatch):
    captured = _capture_broadcasts(monkeypatch)
    cb = ws.TrainingProgressCallback("run-x", output_dir=str(tmp_path))

    asyncio.run(
        cb.on_progress(
            {
                "step": 1,
                "loss": 0.5,
                "epoch": 0,
                "learning_rate": 2e-4,
                "grad_norm": 1.2,
                "eval_loss": 0.61,
                "samples_processed": 8,
                "session_distillation_loss": 0.42,
                "session_distillation_kl": 0.10,
                "session_distillation_ce": 0.31,
                "session_distillation_masked_tokens": 12,
            }
        )
    )

    # Broadcast payload must carry the previously-dropped fields.
    payload = captured[0].payload
    assert payload["eval_loss"] == 0.61
    assert payload["session_distillation_loss"] == 0.42
    assert payload["session_distillation_masked_tokens"] == 12

    # metrics.jsonl must persist them too (so run-analysis gates see real data).
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    assert row["eval_loss"] == 0.61
    assert row["session_distillation_masked_tokens"] == 12


def test_on_progress_omits_absent_optional_metrics(tmp_path, monkeypatch):
    captured = _capture_broadcasts(monkeypatch)
    cb = ws.TrainingProgressCallback("run-y", output_dir=str(tmp_path))

    asyncio.run(cb.on_progress({"step": 1, "loss": 0.5, "epoch": 0, "learning_rate": 2e-4}))

    payload = captured[0].payload
    # No session-distillation on a plain SFT step — don't inject null noise.
    assert "session_distillation_loss" not in payload
