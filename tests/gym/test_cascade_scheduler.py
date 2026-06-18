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
    RepoCascadeDomain,
    _find_stage_checkpoint,
    _strategy_dataset_mismatch,
    build_repo_domains,
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


# =========================================================================
# Phase 4: Repo-Based Domains
# =========================================================================


class TestRepoCascadeDomain:
    def test_matches_by_repo_name(self):
        domain = RepoCascadeDomain(
            name="repo_bashgym",
            description="test",
            reward_mode="verification",
            tool_filter=[],
            min_steps=1,
            repo_names=["bashgym"],
        )
        example = {
            "metadata": {"primary_repo": {"name": "bashgym"}},
            "trace": [{"tool_name": "Bash", "command": "ls"}],
        }
        assert domain.matches(example)

    def test_no_match_wrong_repo(self):
        domain = RepoCascadeDomain(
            name="repo_bashgym",
            description="test",
            reward_mode="verification",
            tool_filter=[],
            min_steps=1,
            repo_names=["bashgym"],
        )
        example = {
            "metadata": {"primary_repo": {"name": "other-repo"}},
            "trace": [{"tool_name": "Bash", "command": "ls"}],
        }
        assert not domain.matches(example)

    def test_min_steps_enforced(self):
        domain = RepoCascadeDomain(
            name="repo_bashgym",
            description="test",
            reward_mode="verification",
            tool_filter=[],
            min_steps=3,
            repo_names=["bashgym"],
        )
        example = {
            "metadata": {"primary_repo": {"name": "bashgym"}},
            "trace": [{"tool_name": "Bash"}],  # Only 1 step
        }
        assert not domain.matches(example)

    def test_matches_messages_format(self):
        domain = RepoCascadeDomain(
            name="repo_test",
            description="test",
            reward_mode="verification",
            tool_filter=[],
            min_steps=1,
            repo_names=["myrepo"],
        )
        example = {
            "metadata": {"primary_repo": {"name": "myrepo"}},
            "messages": [{"role": "assistant", "content": "done"}],
        }
        assert domain.matches(example)


class TestBuildRepoDomains:
    def test_builds_from_gold_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_dir = Path(tmpdir)
            # Create 12 traces for repo-a, 3 for repo-b (below threshold)
            for i in range(12):
                trace = {"metadata": {"primary_repo": {"name": "repo-a"}}, "trace": []}
                (gold_dir / f"trace_{i}.json").write_text(json.dumps(trace))
            for i in range(3):
                trace = {"metadata": {"primary_repo": {"name": "repo-b"}}, "trace": []}
                (gold_dir / f"small_{i}.json").write_text(json.dumps(trace))

            domains = build_repo_domains(gold_dir, min_examples=10)

        assert "repo_repo-a" in domains
        assert "repo_repo-b" not in domains  # below threshold
        assert domains["repo_repo-a"].repo_names == ["repo-a"]

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            domains = build_repo_domains(Path(tmpdir))
        assert domains == {}


class TestCascadeConfigRepoDomains:
    def test_new_fields_default(self):
        config = CascadeConfig()
        assert config.repo_domains_enabled is False
        assert config.repo_domains_dir == ""
        assert config.weakest_first is False


class TestSortDomainsByLoss:
    def test_weakest_first_ordering(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            output_dir=Path(tempfile.mkdtemp()),
        )
        scheduler = CascadeScheduler(config)

        # file_operations has low loss (strong), bash_commands has high loss (weak)
        val_examples = [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "trace": [{"tool_name": "Read"}],
            },
            {
                "messages": [{"role": "user", "content": "run"}],
                "trace": [{"tool_name": "Bash"}],
            },
        ]

        def mock_loss_fn(messages):
            content = messages[0].get("content", "")
            return 1.0 if "hi" in content else 5.0  # bash is "weaker"

        scheduler.sort_domains_by_loss(val_examples, loss_fn=mock_loss_fn)

        # After sorting, bash_commands (higher loss) should be first
        assert scheduler.stages[0].domain.name == "bash_commands"
        assert scheduler.stages[1].domain.name == "file_operations"
        # Stage numbers renumbered
        assert scheduler.stages[0].stage_number == 1
        assert scheduler.stages[1].stage_number == 2

    def test_no_loss_fn_keeps_order(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            output_dir=Path(tempfile.mkdtemp()),
        )
        scheduler = CascadeScheduler(config)
        original_order = [s.domain.name for s in scheduler.stages]
        scheduler.sort_domains_by_loss([], loss_fn=None)
        assert [s.domain.name for s in scheduler.stages] == original_order


# =========================================================================
# Workstream 1: Multi-strategy cascade dispatch
# =========================================================================


class TestStageStrategies:
    def test_default_is_grpo(self):
        config = CascadeConfig(domains=["file_operations", "bash_commands"])
        scheduler = CascadeScheduler(config)
        assert scheduler.stages[0].strategy == "grpo"
        assert scheduler.stages[1].strategy == "grpo"

    def test_explicit_strategies_assigned_in_order(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            stage_strategies=["sft", "dpo"],
        )
        scheduler = CascadeScheduler(config)
        assert scheduler.stages[0].strategy == "sft"
        assert scheduler.stages[1].strategy == "dpo"

    def test_length_mismatch_raises(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            stage_strategies=["sft"],  # too short
        )
        with pytest.raises(ValueError, match="length"):
            CascadeScheduler(config)

    def test_unknown_strategy_raises(self):
        config = CascadeConfig(
            domains=["file_operations"],
            stage_strategies=["rlhf"],  # unknown
        )
        with pytest.raises(ValueError, match="Unknown training strategy"):
            CascadeScheduler(config)

    def test_to_dict_includes_strategy(self):
        config = CascadeConfig(
            domains=["file_operations"],
            stage_strategies=["sft"],
        )
        scheduler = CascadeScheduler(config)
        d = scheduler.stages[0].to_dict()
        assert d["strategy"] == "sft"


class TestStrategyDatasetMismatch:
    def test_sft_accepts_messages(self):
        ex = {"messages": [{"role": "user", "content": "hi"}]}
        assert _strategy_dataset_mismatch("sft", "", ex) is None

    def test_sft_accepts_prompt_completion(self):
        ex = {"prompt": "p", "completion": "c"}
        assert _strategy_dataset_mismatch("sft", "", ex) is None

    def test_sft_rejects_bare_trace(self):
        ex = {"trace": [{"tool_name": "Read"}]}
        msg = _strategy_dataset_mismatch("sft", "", ex)
        assert msg is not None
        assert "messages" in msg or "prompt" in msg

    def test_dpo_accepts_chosen_rejected(self):
        ex = {"chosen": "good", "rejected": "bad"}
        assert _strategy_dataset_mismatch("dpo", "", ex) is None

    def test_dpo_rejects_missing_pairs(self):
        ex = {"messages": [{"role": "user", "content": "hi"}]}
        msg = _strategy_dataset_mismatch("dpo", "", ex)
        assert msg is not None
        assert "chosen" in msg and "rejected" in msg
        assert "pair_failures_for_dpo" in msg

    def test_grpo_defers_to_reward_mode(self):
        # No tests field, verification reward → mismatch
        ex = {"prompt": "p"}
        msg = _strategy_dataset_mismatch("grpo", "verification", ex)
        assert msg is not None
        assert "tests" in msg

    def test_grpo_syntax_needs_prompt_only(self):
        ex = {"prompt": "p"}
        assert _strategy_dataset_mismatch("grpo", "syntax", ex) is None


class TestFindStageCheckpoint:
    def test_prefers_merged_over_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "final").mkdir()
            (root / "merged").mkdir()
            result = _find_stage_checkpoint(root)
            assert result == root / "merged"

    def test_falls_back_to_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "final").mkdir()
            result = _find_stage_checkpoint(root)
            assert result == root / "final"

    def test_walks_nested_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run-abc" / "merged").mkdir(parents=True)
            result = _find_stage_checkpoint(root)
            assert result == root / "run-abc" / "merged"

    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert _find_stage_checkpoint(Path(tmp)) is None

    def test_nonexistent_path(self):
        assert _find_stage_checkpoint(Path("/nonexistent/path")) is None
