# tests/factory/test_synthetic_integration.py
"""Integration tests for the full synthetic data generation pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bashgym.factory.pattern_extractor import PatternExtractor
from bashgym.factory.synthetic_generator import SyntheticGenerator, PRESETS


@pytest.mark.asyncio
async def test_full_synthetic_pipeline():
    """Integration test: extract patterns → generate tasks → export."""

    # 1. Create mock traces
    traces = [
        {
            "metadata": {"repo": "ghostwork"},
            "summary": {"task_description": "Add retry logic to API"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/api.py"}},
                {"tool": "Edit", "input": {"file_path": "src/api.py"}},
                {"tool": "Bash", "input": {"command": "pytest"}},
            ]
        },
        {
            "metadata": {"repo": "ghostwork"},
            "summary": {"task_description": "Fix authentication bug"},
            "trace": [
                {"tool": "Glob", "input": {"pattern": "src/*.py"}},
                {"tool": "Read", "input": {"file_path": "src/auth.py"}},
                {"tool": "Edit", "input": {"file_path": "src/auth.py"}},
            ]
        },
    ]

    # 2. Extract patterns
    extractor = PatternExtractor()
    patterns = extractor.extract_patterns(traces, repo_name="ghostwork")

    assert patterns.repo_name == "ghostwork"
    assert len(patterns.task_types) > 0
    assert "python" in patterns.languages

    # 3. Generate synthetic tasks
    generator = SyntheticGenerator()

    seed_prompts = [t["summary"]["task_description"] for t in traces]

    with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            "Add caching to database queries",
            "Implement rate limiting for API",
            "Fix memory leak in worker",
            "Add validation to user input",
            "Refactor error handling",
        ]

        tasks = await generator.generate_batch(
            patterns=patterns,
            seed_prompts=seed_prompts,
            count=5,
            provider="nim"
        )

        assert len(tasks) == 5
        assert all(t.repo == "ghostwork" for t in tasks)

    # 4. Export to NeMo format
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "synthetic_run"

        metadata = generator.export_to_nemo(tasks, output_dir, train_ratio=0.8)

        assert metadata["total_examples"] == 5
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()

        # Verify JSONL format
        with open(output_dir / "train.jsonl") as f:
            for line in f:
                data = json.loads(line)
                assert "messages" in data
                assert len(data["messages"]) == 3
                assert data["metadata"]["synthetic"] is True


@pytest.mark.asyncio
async def test_pipeline_with_augmented_strategy():
    """Integration test: augmented strategy generates variations of seeds."""

    generator = SyntheticGenerator()

    seed_examples = [
        {"prompt": "Add logging to the API client", "id": "seed_001", "repo": "myrepo"},
        {"prompt": "Fix authentication timeout", "id": "seed_002", "repo": "myrepo"},
    ]

    with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            "Add comprehensive logging with log levels",
            "Add logging to the database module",
            "Fix authentication session expiry",
            "Fix login timeout on slow connections",
        ]

        tasks = await generator.generate_augmented(
            seed_examples=seed_examples,
            variations_per_seed=2,
            provider="nim"
        )

        assert len(tasks) == 4
        assert all(t.metadata.get("strategy") == "augmented" for t in tasks)

    # Export and verify
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "augmented_run"

        metadata = generator.export_to_nemo(tasks, output_dir)

        assert metadata["total_examples"] == 4
        assert (output_dir / "train.jsonl").exists()


@pytest.mark.asyncio
async def test_pipeline_with_schema_driven_strategy():
    """Integration test: schema-driven strategy generates from structure."""

    generator = SyntheticGenerator()

    repo_schema = {
        "name": "test-project",
        "structure": {
            "src/": ["main.py", "utils.py", "config.py"],
            "tests/": ["test_main.py", "test_utils.py"],
        },
        "frameworks": ["fastapi", "pytest"]
    }

    with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            "Add configuration validation",
            "Create utility function for date parsing",
            "Implement main entry point error handling",
        ]

        tasks = await generator.generate_from_schema(
            repo_schema=repo_schema,
            count=3,
            provider="nim"
        )

        assert len(tasks) == 3
        assert all(t.repo == "test-project" for t in tasks)
        assert all(t.metadata.get("strategy") == "schema_driven" for t in tasks)


@pytest.mark.asyncio
async def test_multiplier_calculation_with_presets():
    """Test that presets correctly calculate multipliers."""

    generator = SyntheticGenerator()

    # 84 seed traces (typical real-world count)
    seed_count = 84

    # Quick test preset
    quick_test_multiplier = generator.calculate_multiplier(
        seed_count, PRESETS["quick_test"].target_examples
    )
    assert quick_test_multiplier == 2  # ceil(100/84)

    # Balanced preset
    balanced_multiplier = generator.calculate_multiplier(
        seed_count, PRESETS["balanced"].target_examples
    )
    assert balanced_multiplier == 6  # ceil(500/84)

    # Production preset
    production_multiplier = generator.calculate_multiplier(
        seed_count, PRESETS["production"].target_examples
    )
    assert production_multiplier == 24  # ceil(2000/84)


def test_pattern_extraction_produces_valid_patterns():
    """Test that pattern extraction works with realistic traces."""

    traces = [
        {
            "metadata": {"repo": "webapp"},
            "summary": {"task_description": "Implement user profile page"},
            "trace": [
                {"tool": "Glob", "input": {"pattern": "src/components/*.tsx"}},
                {"tool": "Read", "input": {"file_path": "src/components/Profile.tsx"}},
                {"tool": "Edit", "input": {"file_path": "src/components/Profile.tsx"}},
                {"tool": "Read", "input": {"file_path": "src/api/users.ts"}},
                {"tool": "Edit", "input": {"file_path": "src/api/users.ts"}},
                {"tool": "Bash", "input": {"command": "npm test"}},
            ]
        },
        {
            "metadata": {"repo": "webapp"},
            "summary": {"task_description": "Fix profile image upload"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/components/Profile.tsx"}},
                {"tool": "Edit", "input": {"file_path": "src/components/Profile.tsx"}},
                {"tool": "Bash", "input": {"command": "npm run lint"}},
            ]
        },
        {
            "metadata": {"repo": "webapp"},
            "summary": {"task_description": "Refactor auth module"},
            "trace": [
                {"tool": "Glob", "input": {"pattern": "src/auth/*.ts"}},
                {"tool": "Read", "input": {"file_path": "src/auth/login.ts"}},
                {"tool": "Read", "input": {"file_path": "src/auth/session.ts"}},
                {"tool": "Edit", "input": {"file_path": "src/auth/login.ts"}},
                {"tool": "Edit", "input": {"file_path": "src/auth/session.ts"}},
            ]
        },
    ]

    extractor = PatternExtractor()
    patterns = extractor.extract_patterns(traces, repo_name="webapp")

    # Check repo context
    assert patterns.repo_name == "webapp"

    # Check language detection
    assert "typescript" in patterns.languages
    assert patterns.languages["typescript"] > 0

    # Check task type distribution
    assert len(patterns.task_types) > 0
    assert any(t in patterns.task_types for t in ["feature", "bugfix", "refactor"])

    # Check tool patterns extracted
    assert len(patterns.tool_patterns) > 0

    # Check common paths extracted
    assert len(patterns.common_paths) > 0


def test_export_nemo_format_compatibility():
    """Test that exported JSONL is compatible with NeMo fine-tuning format."""
    from bashgym.factory.synthetic_generator import SyntheticTask

    generator = SyntheticGenerator()

    tasks = [
        SyntheticTask(
            task_id="test_001",
            prompt="Add feature X",
            target_files=["src/feature.py"],
            task_type="feature",
            expected_tools=["Read", "Edit"],
            source_pattern_id="p1",
            repo="test-repo"
        ),
        SyntheticTask(
            task_id="test_002",
            prompt="Fix bug Y",
            target_files=["src/bug.py"],
            task_type="bugfix",
            expected_tools=["Read", "Edit", "Bash"],
            source_pattern_id="p2",
            repo="test-repo"
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "nemo_export"
        generator.export_to_nemo(tasks, output_dir, train_ratio=1.0)

        with open(output_dir / "train.jsonl") as f:
            lines = f.readlines()

        assert len(lines) == 2

        for line in lines:
            data = json.loads(line)

            # NeMo format requirements
            assert "messages" in data
            messages = data["messages"]

            # Must have system, user, assistant
            assert len(messages) == 3
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"

            # Must have content strings
            assert isinstance(messages[0]["content"], str)
            assert isinstance(messages[1]["content"], str)
            assert isinstance(messages[2]["content"], str)
