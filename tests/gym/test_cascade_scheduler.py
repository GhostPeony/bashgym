"""Tests for Cascade RL Scheduler — sequential domain-by-domain training."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bashgym.gym.cascade_scheduler import (
    DOMAIN_TAXONOMY,
    CascadeConfig,
    CascadeScheduler,
    CascadeStage,
)


class TestDomainTaxonomy:
    def test_has_four_domains(self):
        assert len(DOMAIN_TAXONOMY) == 4

    def test_domain_names(self):
        assert "file_operations" in DOMAIN_TAXONOMY
        assert "bash_commands" in DOMAIN_TAXONOMY
        assert "search_and_navigate" in DOMAIN_TAXONOMY
        assert "multi_step_reasoning" in DOMAIN_TAXONOMY

    def test_domains_have_reward_modes(self):
        for domain in DOMAIN_TAXONOMY.values():
            assert domain.reward_mode in ("syntax", "execution", "verification")

    def test_file_ops_reward_is_syntax(self):
        assert DOMAIN_TAXONOMY["file_operations"].reward_mode == "syntax"

    def test_multi_step_reward_is_verification(self):
        assert DOMAIN_TAXONOMY["multi_step_reasoning"].reward_mode == "verification"


class TestCascadeDomainMatches:
    def test_file_ops_matches_read_tool(self):
        domain = DOMAIN_TAXONOMY["file_operations"]
        example = {
            "messages": [
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant",
                    "content": "Reading...",
                    "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}],
                },
            ]
        }
        assert domain.matches(example)

    def test_file_ops_no_match_bash(self):
        domain = DOMAIN_TAXONOMY["file_operations"]
        example = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Running...",
                    "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
                },
            ]
        }
        assert not domain.matches(example)

    def test_bash_matches_bash_tool(self):
        domain = DOMAIN_TAXONOMY["bash_commands"]
        example = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
                },
            ]
        }
        assert domain.matches(example)

    def test_multi_step_requires_min_steps(self):
        domain = DOMAIN_TAXONOMY["multi_step_reasoning"]
        # Only 2 assistant messages — below min_steps=5
        example = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "step1",
                    "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}],
                },
                {
                    "role": "assistant",
                    "content": "step2",
                    "tool_calls": [{"function": {"name": "Edit", "arguments": "{}"}}],
                },
            ]
        }
        assert not domain.matches(example)

    def test_multi_step_matches_many_steps(self):
        domain = DOMAIN_TAXONOMY["multi_step_reasoning"]
        messages = [
            {
                "role": "assistant",
                "content": f"step{i}",
                "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
            }
            for i in range(6)
        ]
        assert domain.matches({"messages": messages})

    def test_empty_messages(self):
        domain = DOMAIN_TAXONOMY["file_operations"]
        assert not domain.matches({"messages": []})
        assert not domain.matches({})


class TestCascadeConfig:
    def test_defaults(self):
        config = CascadeConfig()
        assert len(config.domains) == 4
        assert config.train_steps_per_stage == 200
        assert config.mode == "real"

    def test_custom_domains(self):
        config = CascadeConfig(domains=["file_operations", "bash_commands"])
        assert len(config.domains) == 2

    def test_path_conversion(self):
        config = CascadeConfig(dataset_path="some/path")
        assert isinstance(config.dataset_path, Path)


class TestCascadeStage:
    def test_to_dict(self):
        domain = DOMAIN_TAXONOMY["file_operations"]
        stage = CascadeStage(
            domain=domain,
            stage_number=1,
            base_model="test-model",
            output_path=Path("/tmp/test"),
            run_id="test-001",
        )
        d = stage.to_dict()
        assert d["domain"] == "file_operations"
        assert d["stage_number"] == 1
        assert d["status"] == "pending"


class TestCascadeScheduler:
    def test_init_creates_stages(self):
        config = CascadeConfig(domains=["file_operations", "bash_commands"])
        scheduler = CascadeScheduler(config)
        assert len(scheduler.stages) == 2
        assert scheduler.stages[0].domain.name == "file_operations"
        assert scheduler.stages[1].domain.name == "bash_commands"

    def test_init_unknown_domain_raises(self):
        config = CascadeConfig(domains=["unknown_domain"])
        with pytest.raises(ValueError, match="Unknown domain"):
            CascadeScheduler(config)

    def test_status_idle_initially(self):
        config = CascadeConfig()
        scheduler = CascadeScheduler(config)
        status = scheduler.get_status()
        assert status["status"] == "idle"
        assert status["total_stages"] == 4

    def test_stop(self):
        config = CascadeConfig()
        scheduler = CascadeScheduler(config)
        scheduler.stop()
        assert not scheduler._running


class TestFilterDataset:
    def test_filters_by_domain(self):
        config = CascadeConfig(output_dir=Path(tempfile.mkdtemp()))
        scheduler = CascadeScheduler(config)

        # Create test JSONL with mixed examples
        test_data = Path(tempfile.mktemp(suffix=".jsonl"))
        examples = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "Grep", "arguments": "{}"}}],
                    }
                ]
            },
        ]
        with open(test_data, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        scheduler.config.dataset_path = test_data

        # Filter for file_operations — should get 1 example (Read)
        filtered = scheduler._filter_dataset(DOMAIN_TAXONOMY["file_operations"])
        count = CascadeScheduler._count_jsonl(filtered)
        assert count == 1

    def test_count_jsonl_empty(self):
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        path.write_text("")
        assert CascadeScheduler._count_jsonl(path) == 0

    def test_count_jsonl_nonexistent(self):
        assert CascadeScheduler._count_jsonl(Path("/nonexistent")) == 0


@pytest.mark.asyncio
class TestCascadeRun:
    async def test_simulate_mode_completes(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            mode="simulate",
            output_dir=Path(tempfile.mkdtemp()),
            min_domain_examples=0,  # Don't skip in simulate
        )
        scheduler = CascadeScheduler(config)

        # Create minimal dataset
        test_data = Path(tempfile.mktemp(suffix=".jsonl"))
        test_data.write_text('{"messages": []}\n')
        scheduler.config.dataset_path = test_data

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await scheduler.run_cascade()
        assert result.status in ("completed", "stopped")
        assert len(result.stages) == 2

    async def test_callback_invoked(self):
        events = []

        async def callback(event_type, stage):
            events.append(event_type)

        config = CascadeConfig(
            domains=["file_operations"],
            mode="simulate",
            output_dir=Path(tempfile.mkdtemp()),
            min_domain_examples=0,
        )
        scheduler = CascadeScheduler(config)
        test_data = Path(tempfile.mktemp(suffix=".jsonl"))
        test_data.write_text('{"messages": []}\n')
        scheduler.config.dataset_path = test_data

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await scheduler.run_cascade(callback=callback)
        assert len(events) > 0

    async def test_stop_mid_cascade(self):
        config = CascadeConfig(
            domains=[
                "file_operations",
                "bash_commands",
                "search_and_navigate",
                "multi_step_reasoning",
            ],
            mode="simulate",
            output_dir=Path(tempfile.mkdtemp()),
            min_domain_examples=0,
        )
        scheduler = CascadeScheduler(config)
        test_data = Path(tempfile.mktemp(suffix=".jsonl"))
        test_data.write_text('{"messages": []}\n')
        scheduler.config.dataset_path = test_data

        async def stop_after_first(event_type, stage):
            if event_type == "stage-completed":
                scheduler.stop()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await scheduler.run_cascade(callback=stop_after_first)
        assert result.status == "stopped"
        completed = sum(1 for s in result.stages if s.status == "completed")
        assert completed >= 1
