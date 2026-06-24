"""Tests for the pure remote-hardware preflight parsers.

These cover the GB10/unified-memory case (ponyo): nvidia-smi reports VRAM as
[N/A] on unified memory, so the effective training budget must fall back to
system RAM.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bashgym.gym.remote_trainer import (
    PreflightResult,
    RemoteTrainer,
    SSHConfig,
    parse_meminfo_gb,
    parse_nvidia_smi_gpus,
    remote_compute_budget_gb,
)


def test_parse_nvidia_smi_gpus_discrete():
    out = "NVIDIA RTX 4090, 24576, 23040\nNVIDIA RTX 4090, 24576, 100\n"

    gpus = parse_nvidia_smi_gpus(out)

    assert gpus == [
        {"name": "NVIDIA RTX 4090", "vram_total_gb": 24.0, "vram_free_gb": 22.5},
        {"name": "NVIDIA RTX 4090", "vram_total_gb": 24.0, "vram_free_gb": 0.1},
    ]


def test_parse_nvidia_smi_gpus_handles_na_on_unified_memory():
    # GB10 / Spark unified memory: nvidia-smi cannot report a discrete VRAM total
    gpus = parse_nvidia_smi_gpus("NVIDIA GB10, [N/A], [N/A]\n")

    assert gpus == [{"name": "NVIDIA GB10", "vram_total_gb": None, "vram_free_gb": None}]


def test_parse_meminfo_gb():
    assert parse_meminfo_gb("MemTotal:       131923456 kB\n") == pytest.approx(125.8, abs=0.3)


def test_compute_budget_prefers_discrete_vram():
    budget = remote_compute_budget_gb(
        gpus=[{"name": "x", "vram_total_gb": 24.0, "vram_free_gb": 20.0}],
        ram_gb=64.0,
    )

    assert budget["effective_vram_gb"] == 24.0
    assert budget["unified_memory"] is False


def test_compute_budget_falls_back_to_ram_on_unified_memory():
    budget = remote_compute_budget_gb(
        gpus=[{"name": "NVIDIA GB10", "vram_total_gb": None, "vram_free_gb": None}],
        ram_gb=128.0,
    )

    assert budget["effective_vram_gb"] == 128.0
    assert budget["unified_memory"] is True


def _trainer():
    return RemoteTrainer(
        SSHConfig(
            host="192.168.1.100",
            username="ponyo",
            port=22,
            key_path="~/.ssh/id_rsa",
            remote_work_dir="~/bashgym-training",
        )
    )


def test_preflight_without_unsloth_requirement_stays_ok_for_plain_backend():
    # ponyo (sm_121/GB10) uses the plain transformers backend; a missing Unsloth
    # must not fail preflight when the caller does not require it.
    mock_conn = AsyncMock()
    mock_conn.run = AsyncMock(
        side_effect=[
            MagicMock(stdout="Python 3.12.0\n", exit_status=0),  # python
            MagicMock(stdout="ModuleNotFoundError", exit_status=1),  # unsloth import fails
            MagicMock(stdout="200G\n", exit_status=0),  # disk
        ]
    )

    trainer = _trainer()
    with patch.object(trainer, "_connect", return_value=mock_conn):
        result = asyncio.run(trainer.preflight_check(require_unsloth=False))

    assert result.ok is True
    assert result.unsloth_available is False
    assert any("unsloth" in w.lower() for w in result.warnings)


def test_preflight_capabilities_persists_budget_fields_and_drops_none():
    result = PreflightResult(
        ok=True,
        python_version="Python 3.12.0",
        gpus=[{"name": "NVIDIA GB10", "vram_total_gb": None, "vram_free_gb": None}],
        ram_gb=128.0,
        effective_vram_gb=128.0,
        unified_memory=True,
        unsloth_available=False,
    )

    caps = result.capabilities()

    assert caps["effective_vram_gb"] == 128.0
    assert caps["unified_memory"] is True
    assert caps["unsloth_available"] is False
    assert caps["ram_gb"] == 128.0
    # None-valued discovery fields are dropped so they don't overwrite known data
    assert "disk_free_gb" not in caps
    assert "error" not in caps


def test_preflight_to_dict_round_trips_all_fields():
    result = PreflightResult(ok=True, effective_vram_gb=24.0, unified_memory=False)

    payload = result.to_dict()

    assert payload["ok"] is True
    assert payload["effective_vram_gb"] == 24.0
    assert payload["unified_memory"] is False
    assert "warnings" in payload
