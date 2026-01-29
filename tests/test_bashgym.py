#!/usr/bin/env python3
"""
Comprehensive Test Suite for Bash Gym

Tests all major components:
- Arena (Sandbox, AgentRunner)
- Judge (Verifier)
- Factory (DataFactory, TraceProcessor)
- Gym (Trainer, GymEnv, ModelRouter)

Run with: pytest test_bashgym.py -v
"""

import os
import sys
import json
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
from sandbox import SandboxManager, SandboxConfig, SandboxInstance
from agent_runner import AgentRunner, AgentConfig, TaskResult
from verifier import Verifier, VerificationConfig, VerificationResult, VerificationStatus
from trace_processor import TraceProcessor, ProcessedTrace, TraceQualityMetrics
from bashgym.factory.quality_calculator import (
    calculate_quality_breakdown,
    calculate_complexity,
    calculate_length_score,
    calculate_tool_diversity,
    calculate_efficiency,
    calculate_success_rate,
    QualityBreakdown,
)
from data_factory import DataFactory, DataFactoryConfig, TrainingExample, DPOExample, SynthesisStrategy
from trainer import Trainer, TrainerConfig, TrainingStrategy, TrainingRun, GRPOTrainer
from gym_env import BashGymEnv, GymEnvConfig, Action, ActionType, Observation, BatchGymEnv
from model_router import ModelRouter, RouterConfig, RoutingStrategy, ModelConfig, ModelType
from settings import Settings, get_settings, APISettings, TrainingSettings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_trace_data():
    """Sample trace data for testing."""
    return {
        "metadata": {
            "task_id": "test_task_001",
            "user_initial_prompt": "Create a hello world script",
            "verification_passed": True,
            "started_at": "2024-01-01T00:00:00Z"
        },
        "trace": [
            {
                "tool_name": "Bash",
                "command": "echo 'print(\"Hello, World!\")' > hello.py",
                "output": "",
                "success": True
            },
            {
                "tool_name": "Bash",
                "command": "python hello.py",
                "output": "Hello, World!",
                "success": True
            },
            {
                "tool_name": "Bash",
                "command": "cat hello.py",
                "output": "print(\"Hello, World!\")",
                "success": True
            }
        ],
        "summary": {
            "success_rate": 1.0,
            "total_steps": 3
        }
    }


@pytest.fixture
def sample_trace_file(temp_dir, sample_trace_data):
    """Create a sample trace file."""
    trace_path = temp_dir / "gold_traces" / "test_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(sample_trace_data))
    return trace_path


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for sandbox tests."""
    with patch('sandbox.docker') as mock_docker:
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        # Mock container
        mock_container = MagicMock()
        mock_container.id = "test_container_123"
        mock_container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=(b"output", b"")
        )
        mock_client.containers.create.return_value = mock_container

        yield mock_client


# ============================================================================
# Test: Settings
# ============================================================================

class TestSettings:
    """Tests for the Settings module."""

    def test_settings_creation(self):
        """Test that settings can be created."""
        settings = Settings()
        assert settings is not None
        assert settings.environment is not None

    def test_api_settings_defaults(self):
        """Test API settings have defaults."""
        api = APISettings()
        assert api.anthropic_model == "claude-sonnet-4-20250514"
        assert api.nim_endpoint == "https://integrate.api.nvidia.com/v1"

    def test_training_settings_defaults(self):
        """Test training settings have sensible defaults."""
        training = TrainingSettings()
        assert training.base_model == "meta-llama/Llama-3.2-3B-Instruct"
        assert training.learning_rate == 2e-5
        assert training.use_lora == True
        assert training.lora_r == 16

    def test_settings_to_dict_hides_secrets(self):
        """Test that to_dict hides sensitive values."""
        settings = Settings()
        settings.api.anthropic_api_key = "secret_key_123"

        result = settings.to_dict()
        assert result["api"]["anthropic_api_key"] == "***"

    def test_get_settings_singleton(self):
        """Test that get_settings returns same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


# ============================================================================
# Test: Sandbox Manager
# ============================================================================

class TestSandboxManager:
    """Tests for the Sandbox Manager (Arena)."""

    def test_sandbox_config_defaults(self):
        """Test sandbox config has sensible defaults."""
        config = SandboxConfig()
        assert config.image == "python:3.10-slim"
        assert config.memory_limit == "2g"
        assert config.network_mode == "none"

    def test_dangerous_command_detection(self, mock_docker_client):
        """Test that dangerous commands are blocked."""
        manager = SandboxManager()

        dangerous_commands = [
            "rm -rf /",
            "rm -rf /*",
            "mkfs.ext4 /dev/sda",
            ":(){:|:&};:",  # Fork bomb
        ]

        for cmd in dangerous_commands:
            assert manager._is_dangerous_command(cmd), f"Should block: {cmd}"

    def test_safe_commands_allowed(self, mock_docker_client):
        """Test that safe commands are allowed."""
        manager = SandboxManager()

        safe_commands = [
            "ls -la",
            "cat file.txt",
            "python script.py",
            "pip install requests",
            "git status",
        ]

        for cmd in safe_commands:
            assert not manager._is_dangerous_command(cmd), f"Should allow: {cmd}"


# ============================================================================
# Test: Verifier
# ============================================================================

class TestVerifier:
    """Tests for the Verifier (Judge)."""

    def test_verifier_config_defaults(self):
        """Test verifier config defaults."""
        config = VerificationConfig()
        assert config.timeout == 300
        assert "test_*.py" in config.test_patterns

    def test_verification_result_success_property(self):
        """Test VerificationResult success property."""
        result = VerificationResult(
            task_id="test",
            status=VerificationStatus.PASSED,
            exit_code=0,
            passed_tests=5,
            failed_tests=0,
            total_tests=5,
            duration_seconds=1.0,
            stdout="",
            stderr=""
        )
        assert result.success == True

        result.status = VerificationStatus.FAILED
        assert result.success == False

    def test_verification_result_to_dict(self):
        """Test VerificationResult serialization."""
        result = VerificationResult(
            task_id="test_123",
            status=VerificationStatus.PASSED,
            exit_code=0,
            passed_tests=3,
            failed_tests=0,
            total_tests=3,
            duration_seconds=2.5,
            stdout="All tests passed",
            stderr=""
        )

        data = result.to_dict()
        assert data["task_id"] == "test_123"
        assert data["status"] == "passed"
        assert data["passed_tests"] == 3

    def test_find_test_files(self, temp_dir):
        """Test finding test files in workspace."""
        # Create test files
        (temp_dir / "test_example.py").write_text("def test_foo(): pass")
        (temp_dir / "utils.py").write_text("# not a test")
        (temp_dir / "tests").mkdir()
        (temp_dir / "tests" / "test_more.py").write_text("def test_bar(): pass")

        verifier = Verifier()
        test_files = verifier._find_test_files(temp_dir)

        assert len(test_files) >= 1
        assert any("test_example.py" in str(f) for f in test_files)

    def test_find_verify_script(self, temp_dir):
        """Test finding verification scripts."""
        verifier = Verifier()

        # No script
        assert verifier._find_verify_script(temp_dir) is None

        # Create verify.sh
        verify_script = temp_dir / "verify.sh"
        verify_script.write_text("#!/bin/bash\nexit 0")

        found = verifier._find_verify_script(temp_dir)
        assert found is not None
        assert found.name == "verify.sh"


# ============================================================================
# Test: Trace Processor
# ============================================================================

class TestTraceProcessor:
    """Tests for the Trace Processor (Factory)."""

    def test_quality_metrics_calculation(self):
        """Test TraceQualityMetrics calculations."""
        metrics = TraceQualityMetrics(
            total_steps=10,
            successful_steps=8,
            failed_steps=2,
            unique_commands=5,
            verification_passed=True,
            complexity_score=5.0
        )

        assert metrics.success_rate == 0.8
        assert metrics.quality_score > 0.5  # Should be decent quality

    def test_process_trace(self, sample_trace_file):
        """Test processing a trace file."""
        processor = TraceProcessor(min_quality_score=0.1)

        result = processor.process_trace(sample_trace_file)

        assert result is not None
        assert result.task_prompt == "Create a hello world script"
        assert len(result.normalized_steps) == 3
        assert result.quality_metrics.verification_passed == True

    def test_sensitive_data_redaction(self):
        """Test that sensitive data is redacted."""
        processor = TraceProcessor()

        sensitive_texts = [
            "API_KEY=sk-abc123xyz",
            "Bearer nvapi-secret-token",
            "password: mysecret123",
        ]

        for text in sensitive_texts:
            redacted = processor._redact_sensitive(text)
            assert "[REDACTED]" in redacted

    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        processor = TraceProcessor()

        simple_steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "bash", "command": "cat file.txt"}
        ]

        complex_steps = [
            {"tool": "bash", "command": "find . -name '*.py' | xargs grep 'def'"},
            {"tool": "bash", "command": "if [ -f test.py ]; then python test.py; fi"},
            {"tool": "write", "command": "complex_script.py"},
            {"tool": "bash", "command": "docker build && docker run"}
        ]

        simple_score = processor._calculate_complexity(simple_steps)
        complex_score = processor._calculate_complexity(complex_steps)

        assert complex_score > simple_score

    def test_deduplication(self, temp_dir, sample_trace_data):
        """Test that duplicate traces are filtered."""
        processor = TraceProcessor(deduplicate=True)

        # Create two identical traces
        trace1 = temp_dir / "trace1.json"
        trace2 = temp_dir / "trace2.json"
        trace1.write_text(json.dumps(sample_trace_data))
        trace2.write_text(json.dumps(sample_trace_data))

        result1 = processor.process_trace(trace1)
        result2 = processor.process_trace(trace2)

        assert result1 is not None
        assert result2 is None  # Should be filtered as duplicate


# ============================================================================
# Test: Quality Calculator
# ============================================================================

class TestQualityCalculator:
    """Tests for the centralized Quality Calculator module."""

    def test_calculate_success_rate_all_success(self):
        """Test success rate with all successful steps."""
        steps = [
            {"success": True, "exit_code": 0},
            {"success": True, "exit_code": 0},
            {"success": True, "exit_code": 0},
        ]
        rate, successful, failed = calculate_success_rate(steps)
        assert rate == 1.0
        assert successful == 3
        assert failed == 0

    def test_calculate_success_rate_mixed(self):
        """Test success rate with mixed results."""
        steps = [
            {"success": True},
            {"success": False},
            {"exit_code": 0},
            {"exit_code": 1},
        ]
        rate, successful, failed = calculate_success_rate(steps)
        assert rate == 0.5
        assert successful == 2
        assert failed == 2

    def test_calculate_success_rate_empty(self):
        """Test success rate with empty steps."""
        rate, successful, failed = calculate_success_rate([])
        assert rate == 0.0
        assert successful == 0
        assert failed == 0

    def test_calculate_complexity_simple(self):
        """Test complexity for simple traces."""
        steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "bash", "command": "cat file.txt"},
        ]
        score, unique_cmds = calculate_complexity(steps)
        assert 0.0 <= score <= 1.0
        assert unique_cmds >= 1

    def test_calculate_complexity_diverse_tools(self):
        """Test complexity increases with tool diversity."""
        simple_steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "bash", "command": "pwd"},
        ]

        diverse_steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "read", "command": "file.txt"},
            {"tool": "write", "command": "output.txt"},
            {"tool": "grep", "command": "pattern"},
        ]

        simple_score, _ = calculate_complexity(simple_steps)
        diverse_score, _ = calculate_complexity(diverse_steps)

        assert diverse_score > simple_score

    def test_calculate_complexity_with_control_flow(self):
        """Test complexity bonus for control flow patterns."""
        basic_steps = [
            {"tool": "bash", "command": "echo hello"},
        ]

        control_flow_steps = [
            {"tool": "bash", "command": "if [ -f test.py ]; then python test.py; fi"},
            {"tool": "bash", "command": "for f in *.txt; do cat $f; done"},
        ]

        basic_score, _ = calculate_complexity(basic_steps)
        control_flow_score, _ = calculate_complexity(control_flow_steps)

        assert control_flow_score > basic_score

    def test_calculate_length_score_ideal(self):
        """Test length score for ideal trace lengths."""
        # 10-20 steps is ideal (1.0)
        assert calculate_length_score(10) == 1.0
        assert calculate_length_score(15) == 1.0
        assert calculate_length_score(20) == 1.0

    def test_calculate_length_score_bell_curve(self):
        """Test length score follows bell curve."""
        # Too short
        assert calculate_length_score(1) == 0.2
        assert calculate_length_score(2) == 0.2
        assert calculate_length_score(5) == 0.5

        # Good
        assert calculate_length_score(8) == 0.8

        # Ideal
        assert calculate_length_score(15) == 1.0

        # Acceptable
        assert calculate_length_score(25) == 0.8

        # Long
        assert calculate_length_score(40) == 0.5

        # Too long
        assert calculate_length_score(100) == 0.2

    def test_calculate_tool_diversity(self):
        """Test tool diversity scoring."""
        # 1 tool = 0.2
        one_tool = [{"tool": "bash"}]
        score, count = calculate_tool_diversity(one_tool)
        assert score == 0.2
        assert count == 1

        # 2 tools = 0.5
        two_tools = [{"tool": "bash"}, {"tool": "read"}]
        score, count = calculate_tool_diversity(two_tools)
        assert score == 0.5
        assert count == 2

        # 3 tools = 0.75
        three_tools = [{"tool": "bash"}, {"tool": "read"}, {"tool": "write"}]
        score, count = calculate_tool_diversity(three_tools)
        assert score == 0.75
        assert count == 3

        # 4+ tools = 1.0
        four_tools = [{"tool": "bash"}, {"tool": "read"}, {"tool": "write"}, {"tool": "grep"}]
        score, count = calculate_tool_diversity(four_tools)
        assert score == 1.0
        assert count == 4

    def test_calculate_efficiency_perfect(self):
        """Test efficiency score for perfect execution."""
        steps = [
            {"success": True, "output": "result 1"},
            {"success": True, "output": "result 2"},
            {"success": True, "output": "result 3"},
        ]
        score = calculate_efficiency(steps)
        assert score >= 0.8  # High efficiency

    def test_calculate_efficiency_with_recovery(self):
        """Test efficiency score rewards error recovery."""
        # Failure followed by success (recovery)
        steps = [
            {"success": False, "error": "Failed"},
            {"success": True, "output": "Fixed it"},
            {"success": True, "output": "Done"},
        ]
        score = calculate_efficiency(steps)
        assert 0.4 <= score <= 0.9  # Decent score due to recovery

    def test_calculate_efficiency_with_retries(self):
        """Test efficiency score penalizes repeated commands."""
        steps = [
            {"command": "git push", "success": False},
            {"command": "git push", "success": False},  # Retry
            {"command": "git push", "success": True},   # Same command
        ]
        score = calculate_efficiency(steps)
        assert score < 0.8  # Penalized for retries

    def test_calculate_quality_breakdown_complete(self):
        """Test complete quality breakdown calculation."""
        steps = [
            {"tool": "bash", "command": "ls", "success": True, "output": "files"},
            {"tool": "read", "command": "file.txt", "success": True, "output": "content"},
            {"tool": "write", "command": "out.txt", "success": True, "output": "written"},
            {"tool": "bash", "command": "python test.py", "success": True, "output": "PASSED"},
        ]
        metadata = {"verification_passed": True}

        breakdown = calculate_quality_breakdown(steps, metadata=metadata)

        assert isinstance(breakdown, QualityBreakdown)
        assert breakdown.success_rate == 1.0
        assert breakdown.verification_score == 1.0
        assert breakdown.complexity_score > 0
        assert breakdown.length_score > 0
        assert breakdown.tool_diversity > 0
        assert breakdown.efficiency_score > 0
        assert 0.0 <= breakdown.total_score <= 1.0

    def test_calculate_quality_breakdown_weights(self):
        """Test that quality score uses correct weights."""
        # Perfect scores on all metrics should give ~1.0 total
        steps = [
            {"tool": "bash", "command": "cmd1", "success": True, "output": "out"},
            {"tool": "read", "command": "cmd2", "success": True, "output": "out"},
            {"tool": "write", "command": "cmd3", "success": True, "output": "out"},
            {"tool": "grep", "command": "cmd4", "success": True, "output": "out"},
            {"tool": "glob", "command": "cmd5", "success": True, "output": "out"},
            {"tool": "edit", "command": "cmd6", "success": True, "output": "out"},
            {"tool": "bash", "command": "test", "success": True, "output": "PASSED"},
            {"tool": "bash", "command": "commit", "success": True, "output": "done"},
            {"tool": "bash", "command": "push", "success": True, "output": "pushed"},
            {"tool": "bash", "command": "deploy", "success": True, "output": "deployed"},
        ]
        metadata = {"verification_passed": True}

        breakdown = calculate_quality_breakdown(steps, metadata=metadata)

        # Should have high total score with all metrics positive
        assert breakdown.total_score >= 0.7

    def test_quality_breakdown_to_dict(self):
        """Test QualityBreakdown serialization."""
        breakdown = QualityBreakdown(
            success_rate=0.9,
            verification_score=1.0,
            complexity_score=0.7,
            length_score=0.8,
            tool_diversity=0.75,
            efficiency_score=0.85,
            total_score=0.82,
            total_steps=15,
            successful_steps=14,
            failed_steps=1,
            unique_tools_count=4,
            unique_commands_count=10,
        )

        data = breakdown.to_dict()
        assert data["success_rate"] == 0.9
        assert data["verification_score"] == 1.0
        assert data["total_score"] == 0.82


# ============================================================================
# Test: Data Factory
# ============================================================================

class TestDataFactory:
    """Tests for the Data Factory."""

    def test_factory_config_defaults(self):
        """Test factory config defaults."""
        config = DataFactoryConfig()
        assert config.strategy == SynthesisStrategy.AUGMENTED
        assert config.augmentation_factor == 3

    def test_training_example_to_dict(self):
        """Test TrainingExample serialization."""
        example = TrainingExample(
            example_id="ex_001",
            system_prompt="You are a helpful assistant.",
            user_prompt="Write hello world",
            assistant_response="print('Hello, World!')"
        )

        data = example.to_dict()
        assert data["id"] == "ex_001"
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "system"

    def test_training_example_to_chatml(self):
        """Test ChatML format conversion."""
        example = TrainingExample(
            example_id="ex_001",
            system_prompt="System prompt",
            user_prompt="User prompt",
            assistant_response="Assistant response"
        )

        chatml = example.to_chatml()
        assert "<|im_start|>system" in chatml
        assert "<|im_start|>user" in chatml
        assert "<|im_start|>assistant" in chatml
        assert "<|im_end|>" in chatml

    def test_dpo_example_to_dict(self):
        """Test DPOExample serialization."""
        example = DPOExample(
            example_id="dpo_001",
            prompt="Fix the bug",
            chosen="Good solution",
            rejected="Bad solution"
        )

        data = example.to_dict()
        assert data["prompt"] == "Fix the bug"
        assert data["chosen"] == "Good solution"
        assert data["rejected"] == "Bad solution"

    def test_process_gold_trace(self, sample_trace_file):
        """Test processing a gold trace into training example."""
        factory = DataFactory()

        example = factory.process_gold_trace(sample_trace_file)

        assert example is not None
        assert "Create a hello world script" in example.user_prompt
        assert len(example.assistant_response) > 0

    def test_trace_validation(self):
        """Test trace validation logic."""
        factory = DataFactory(DataFactoryConfig(
            min_trace_steps=2,
            max_trace_steps=10,
            require_successful_verification=True
        ))

        # Valid trace
        valid_trace = {
            "metadata": {
                "user_initial_prompt": "Do something",
                "verification_passed": True
            },
            "trace": [{"step": 1}, {"step": 2}, {"step": 3}]
        }
        assert factory._validate_trace(valid_trace) == True

        # Too few steps
        short_trace = {
            "metadata": {"user_initial_prompt": "Do something", "verification_passed": True},
            "trace": [{"step": 1}]
        }
        assert factory._validate_trace(short_trace) == False

        # Verification failed
        failed_trace = {
            "metadata": {"user_initial_prompt": "Do something", "verification_passed": False},
            "trace": [{"step": 1}, {"step": 2}]
        }
        assert factory._validate_trace(failed_trace) == False


# ============================================================================
# Test: Trainer
# ============================================================================

class TestTrainer:
    """Tests for the Trainer (Gym)."""

    def test_trainer_config_defaults(self):
        """Test trainer config defaults."""
        config = TrainerConfig()
        assert config.base_model == "meta-llama/Llama-3.2-3B-Instruct"
        assert config.strategy == TrainingStrategy.SFT
        assert config.use_lora == True

    def test_training_run_to_dict(self):
        """Test TrainingRun serialization."""
        run = TrainingRun(
            run_id="run_001",
            strategy=TrainingStrategy.SFT,
            base_model="llama-3b",
            dataset_path=Path("data/train.jsonl"),
            output_path=Path("models/run_001"),
            status="completed",
            metrics={"loss": 1.5}
        )

        data = run.to_dict()
        assert data["run_id"] == "run_001"
        assert data["strategy"] == "sft"
        assert data["status"] == "completed"

    def test_generate_run_id(self):
        """Test run ID generation."""
        import time
        trainer = Trainer()

        id1 = trainer._generate_run_id()
        time.sleep(0.01)  # Small delay to ensure different timestamp
        id2 = trainer._generate_run_id()

        assert id1.startswith("run_")
        # IDs may be same within same second, just verify format
        assert "_" in id1

    def test_sft_script_generation(self, temp_dir):
        """Test SFT training script generation."""
        config = TrainerConfig(
            base_model="test-model",
            num_epochs=2,
            batch_size=8
        )
        trainer = Trainer(config)

        run = TrainingRun(
            run_id="test_run",
            strategy=TrainingStrategy.SFT,
            base_model="test-model",
            dataset_path=Path("data.jsonl"),
            output_path=temp_dir / "output"
        )

        script = trainer._generate_unsloth_sft_script(run)

        assert "from unsloth import FastLanguageModel" in script
        assert "test-model" in script
        assert "num_train_epochs=2" in script

    def test_dpo_script_generation(self, temp_dir):
        """Test DPO training script generation."""
        config = TrainerConfig(dpo_beta=0.2)
        trainer = Trainer(config)

        run = TrainingRun(
            run_id="dpo_run",
            strategy=TrainingStrategy.DPO,
            base_model="test-model",
            dataset_path=Path("dpo_data.jsonl"),
            output_path=temp_dir / "output"
        )

        script = trainer._generate_unsloth_dpo_script(run)

        assert "DPOTrainer" in script
        assert "beta=0.2" in script


# ============================================================================
# Test: Gym Environment
# ============================================================================

class TestGymEnv:
    """Tests for the Gym Environment."""

    def test_gym_config_defaults(self):
        """Test gym config defaults."""
        config = GymEnvConfig()
        assert config.max_steps == 50
        assert config.success_reward == 1.0
        assert config.step_penalty == -0.01

    def test_action_creation(self):
        """Test Action creation and serialization."""
        action = Action(
            action_type=ActionType.BASH,
            content="ls -la",
            metadata={"cwd": "/workspace"}
        )

        data = action.to_dict()
        assert data["type"] == "bash"
        assert data["content"] == "ls -la"

    def test_observation_creation(self):
        """Test Observation creation and serialization."""
        obs = Observation(
            content="File listing...",
            success=True,
            done=False,
            info={"step": 1}
        )

        data = obs.to_dict()
        assert data["success"] == True
        assert data["done"] == False

    def test_env_reset(self):
        """Test environment reset."""
        config = GymEnvConfig(use_sandbox=False)
        env = BashGymEnv(config)

        obs = env.reset(
            task="Test task",
            task_id="test_001"
        )

        assert obs is not None
        assert "Test task" in obs.content
        assert env.step_count == 0
        assert env.done == False

    def test_env_step(self):
        """Test environment step."""
        config = GymEnvConfig(use_sandbox=False, max_steps=5)
        env = BashGymEnv(config)

        env.reset(task="Test task")

        action = Action(ActionType.BASH, "echo hello")
        obs, reward, done, info = env.step(action)

        assert obs is not None
        assert env.step_count == 1
        assert reward <= 0  # Step penalty

    def test_env_max_steps(self):
        """Test that environment terminates at max steps."""
        config = GymEnvConfig(use_sandbox=False, max_steps=3)
        env = BashGymEnv(config)

        env.reset(task="Test task")

        for i in range(5):
            action = Action(ActionType.BASH, f"echo step {i}")
            obs, reward, done, info = env.step(action)
            if done:
                break

        assert env.done == True
        assert env.step_count <= 3

    def test_env_submit_action(self):
        """Test submit action terminates episode."""
        config = GymEnvConfig(use_sandbox=False)
        env = BashGymEnv(config)

        env.reset(task="Test task")

        action = Action(ActionType.SUBMIT, "")
        obs, reward, done, info = env.step(action)

        assert done == True
        assert obs.info.get("submitted") == True

    def test_trajectory_logging(self, temp_dir):
        """Test trajectory is logged correctly."""
        config = GymEnvConfig(
            use_sandbox=False,
            log_trajectory=True,
            trajectory_dir=str(temp_dir)
        )
        env = BashGymEnv(config)

        env.reset(task="Test task")
        env.step(Action(ActionType.BASH, "echo 1"))
        env.step(Action(ActionType.BASH, "echo 2"))
        env.step(Action(ActionType.SUBMIT, ""))

        trajectory = env.get_trajectory()
        assert len(trajectory) == 4  # Initial + 3 steps

        # Save trajectory
        path = env.save_trajectory()
        assert path.exists()

    def test_batch_env(self):
        """Test batch environment."""
        config = GymEnvConfig(use_sandbox=False, max_steps=5)
        batch_env = BatchGymEnv(num_envs=3, config=config)

        observations = batch_env.reset_all("Test task", "batch")
        assert len(observations) == 3

        actions = [Action(ActionType.BASH, f"echo {i}") for i in range(3)]
        results = batch_env.step_all(actions)
        assert len(results) == 3

        rewards = batch_env.get_rewards()
        assert len(rewards) == 3


# ============================================================================
# Test: Model Router
# ============================================================================

class TestModelRouter:
    """Tests for the Model Router."""

    def test_router_config_defaults(self):
        """Test router config defaults."""
        config = RouterConfig()
        assert config.strategy == RoutingStrategy.CONFIDENCE_BASED
        assert config.confidence_threshold == 0.7

    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            name="test-model",
            model_type=ModelType.TEACHER,
            endpoint="https://api.example.com",
            api_key="test-key"
        )

        assert config.name == "test-model"
        assert config.model_type == ModelType.TEACHER

    def test_register_model(self):
        """Test model registration."""
        router = ModelRouter()

        model = ModelConfig(
            name="custom-model",
            model_type=ModelType.STUDENT,
            endpoint="http://localhost:8000"
        )
        router.register_model(model)

        assert "custom-model" in router.models
        assert router.get_student_model() is not None

    def test_complexity_estimation(self):
        """Test task complexity estimation."""
        router = ModelRouter()

        simple_prompt = "Fix typo in README"
        complex_prompt = """Refactor the authentication system to use OAuth2 with PKCE flow.
        First, update the auth module. Then, modify the database schema.
        After that, implement the token refresh mechanism.
        Finally, add comprehensive security tests."""

        simple_score = router._estimate_complexity(simple_prompt)
        complex_score = router._estimate_complexity(complex_prompt)

        assert complex_score > simple_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1

    def test_routing_decision(self):
        """Test routing decision creation."""
        router = ModelRouter(RouterConfig(strategy=RoutingStrategy.TEACHER_ONLY))

        # Register a teacher model
        router.register_model(ModelConfig(
            name="teacher",
            model_type=ModelType.TEACHER,
            endpoint="https://api.example.com"
        ))

        decision = router.route("Test prompt")

        assert decision is not None
        # Check model type instead of name (name may be "teacher" or loaded from env)
        assert decision.model_type == ModelType.TEACHER

    def test_progressive_routing(self):
        """Test progressive routing increases student usage."""
        config = RouterConfig(
            strategy=RoutingStrategy.PROGRESSIVE,
            student_sample_rate=0.5
        )
        router = ModelRouter(config)

        # Register both models
        router.register_model(ModelConfig("teacher", ModelType.TEACHER, "http://t"))
        router.register_model(ModelConfig("student", ModelType.STUDENT, "http://s"))

        # Make many routing decisions
        student_count = 0
        for _ in range(100):
            decision = router.route("Test prompt")
            if decision.model_type == ModelType.STUDENT:
                student_count += 1

        # Should be roughly 50% student (with some variance)
        assert 30 < student_count < 70

    def test_routing_stats(self):
        """Test routing statistics."""
        router = ModelRouter()
        router.register_model(ModelConfig("teacher", ModelType.TEACHER, "http://t"))

        # Make some routing decisions
        for _ in range(5):
            router.route("Test prompt")

        stats = router.get_routing_stats()
        assert stats["total_requests"] == 5
        assert "teacher_requests" in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_trace_to_training_pipeline(self, temp_dir, sample_trace_data):
        """Test the trace -> process -> training example pipeline."""
        # Create trace file
        trace_path = temp_dir / "trace.json"
        trace_path.write_text(json.dumps(sample_trace_data))

        # Process trace
        processor = TraceProcessor(min_quality_score=0.1)
        processed = processor.process_trace(trace_path)
        assert processed is not None

        # Create training example
        factory = DataFactory()
        example = factory.process_gold_trace(trace_path)
        assert example is not None

        # Verify example format
        data = example.to_dict()
        assert "messages" in data
        assert len(data["messages"]) == 3

    def test_gym_trajectory_to_trace(self, temp_dir):
        """Test gym trajectory can be converted to trace format."""
        config = GymEnvConfig(
            use_sandbox=False,
            trajectory_dir=str(temp_dir)
        )
        env = BashGymEnv(config)

        # Run a simple episode
        env.reset(task="Create hello.py")
        env.step(Action(ActionType.BASH, "echo 'print(1)' > hello.py"))
        env.step(Action(ActionType.BASH, "python hello.py"))
        env.step(Action(ActionType.SUBMIT, ""))

        # Save and verify trajectory
        path = env.save_trajectory()

        with open(path) as f:
            trajectory_data = json.load(f)

        assert "task" in trajectory_data
        assert "trajectory" in trajectory_data
        assert len(trajectory_data["trajectory"]) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
