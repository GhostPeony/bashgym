# tests/factory/test_synthetic_generator.py
"""Tests for synthetic task generation data structures."""

import pytest
from bashgym.factory.synthetic_generator import (
    SyntheticTask,
    GenerationStrategy,
    GenerationPreset,
    PRESETS,
    SyntheticGenerator,
)


class TestSyntheticTask:
    """Tests for SyntheticTask dataclass."""

    def test_synthetic_task_dataclass(self):
        """SyntheticTask should store generated task data."""
        task = SyntheticTask(
            task_id="synth_001",
            prompt="Add logging to the API client",
            target_files=["src/api.py", "tests/test_api.py"],
            task_type="feature",
            expected_tools=["Read", "Edit", "Bash"],
            source_pattern_id="pattern_003",
            repo="ghostwork",
        )

        assert task.task_id == "synth_001"
        assert task.prompt == "Add logging to the API client"
        assert task.task_type == "feature"
        assert len(task.target_files) == 2
        assert task.target_files == ["src/api.py", "tests/test_api.py"]
        assert task.expected_tools == ["Read", "Edit", "Bash"]
        assert task.source_pattern_id == "pattern_003"
        assert task.repo == "ghostwork"

    def test_synthetic_task_default_metadata(self):
        """SyntheticTask should have empty dict as default metadata."""
        task = SyntheticTask(
            task_id="synth_002",
            prompt="Fix bug",
            target_files=["src/bug.py"],
            task_type="bugfix",
            expected_tools=["Read", "Edit"],
            source_pattern_id="pattern_001",
            repo="test-repo",
        )

        assert task.metadata == {}

    def test_synthetic_task_with_metadata(self):
        """SyntheticTask should accept custom metadata."""
        task = SyntheticTask(
            task_id="synth_003",
            prompt="Refactor module",
            target_files=["src/module.py"],
            task_type="refactor",
            expected_tools=["Read", "Edit"],
            source_pattern_id="pattern_002",
            repo="test-repo",
            metadata={"complexity": "high", "estimated_time": 30},
        )

        assert task.metadata["complexity"] == "high"
        assert task.metadata["estimated_time"] == 30


class TestGenerationStrategy:
    """Tests for GenerationStrategy enum."""

    def test_generation_strategy_values(self):
        """GenerationStrategy should have expected enum values."""
        assert GenerationStrategy.TRACE_SEEDED.value == "trace_seeded"
        assert GenerationStrategy.AUGMENTED.value == "augmented"
        assert GenerationStrategy.SCHEMA_DRIVEN.value == "schema_driven"

    def test_generation_strategy_is_string_enum(self):
        """GenerationStrategy should be a string enum."""
        assert isinstance(GenerationStrategy.TRACE_SEEDED, str)
        assert GenerationStrategy.TRACE_SEEDED == "trace_seeded"


class TestGenerationPreset:
    """Tests for GenerationPreset dataclass."""

    def test_generation_preset_dataclass(self):
        """GenerationPreset should store preset configuration."""
        preset = GenerationPreset(
            label="Test Preset",
            description="A test preset for validation",
            target_examples=250,
        )

        assert preset.label == "Test Preset"
        assert preset.description == "A test preset for validation"
        assert preset.target_examples == 250

    def test_generation_preset_default_multiplier(self):
        """GenerationPreset should have None as default multiplier."""
        preset = GenerationPreset(
            label="Default Multiplier",
            description="Testing default multiplier",
            target_examples=100,
        )

        assert preset.multiplier is None

    def test_generation_preset_with_multiplier(self):
        """GenerationPreset should accept custom multiplier."""
        preset = GenerationPreset(
            label="Custom Multiplier",
            description="Testing custom multiplier",
            target_examples=500,
            multiplier=5,
        )

        assert preset.multiplier == 5


class TestPresets:
    """Tests for PRESETS dictionary."""

    def test_presets_exist(self):
        """Should have all required presets defined."""
        assert "quick_test" in PRESETS
        assert "balanced" in PRESETS
        assert "production" in PRESETS
        assert "custom" in PRESETS

    def test_quick_test_preset(self):
        """Quick test preset should have 100 target examples."""
        assert PRESETS["quick_test"].target_examples == 100
        assert PRESETS["quick_test"].label == "Quick Test"

    def test_balanced_preset(self):
        """Balanced preset should have 500 target examples."""
        assert PRESETS["balanced"].target_examples == 500
        assert PRESETS["balanced"].label == "Balanced (Recommended)"

    def test_production_preset(self):
        """Production preset should have 2000 target examples."""
        assert PRESETS["production"].target_examples == 2000
        assert PRESETS["production"].label == "Production"

    def test_custom_preset(self):
        """Custom preset should have None target examples."""
        assert PRESETS["custom"].target_examples is None
        assert PRESETS["custom"].label == "Custom"

    def test_all_presets_are_generation_preset_type(self):
        """All presets should be GenerationPreset instances."""
        for key, preset in PRESETS.items():
            assert isinstance(preset, GenerationPreset), f"{key} is not a GenerationPreset"


class TestSyntheticGenerator:
    """Tests for SyntheticGenerator class."""

    def test_calculate_multiplier(self):
        """Should calculate multiplier from seed count and target."""
        generator = SyntheticGenerator()

        # 84 seeds, want 500 -> ceil(500/84) = 6
        assert generator.calculate_multiplier(seed_count=84, target_examples=500) == 6

        # 84 seeds, want 100 -> ceil(100/84) = 2
        assert generator.calculate_multiplier(seed_count=84, target_examples=100) == 2

        # 84 seeds, want 2000 -> ceil(2000/84) = 24
        assert generator.calculate_multiplier(seed_count=84, target_examples=2000) == 24

    def test_calculate_multiplier_edge_cases(self):
        """Should handle edge cases."""
        generator = SyntheticGenerator()

        # Minimum multiplier of 1
        assert generator.calculate_multiplier(seed_count=1000, target_examples=100) == 1

        # Zero seeds returns 0
        assert generator.calculate_multiplier(seed_count=0, target_examples=500) == 0

    def test_generator_has_presets(self):
        """SyntheticGenerator should have access to presets."""
        generator = SyntheticGenerator()
        assert generator.presets == PRESETS

    def test_build_generation_prompt(self):
        """Should build LLM prompt for task generation."""
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"feature": 0.5, "bugfix": 0.5},
            repo_name="ghostwork",
            framework_hints=["fastapi", "pytest"],
            common_paths=["src/", "bashgym/"],
            languages={"python": 0.8},
            prompt_keywords=["add", "fix", "implement"]
        )

        seed_prompts = [
            "Add retry logic to the API client",
            "Fix the authentication bug",
        ]

        prompt = generator._build_generation_prompt(
            patterns=patterns,
            task_type="feature",
            seed_prompts=seed_prompts
        )

        assert "ghostwork" in prompt
        assert "feature" in prompt
        assert "fastapi" in prompt
        assert "Add retry logic" in prompt

    @pytest.mark.asyncio
    async def test_generate_task_with_nim(self):
        """Should call NIM to generate a synthetic task."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"feature": 1.0},
            repo_name="ghostwork",
            common_paths=["src/"],
            languages={"python": 1.0}
        )

        # Mock the LLM client
        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Add caching to the database queries"

            task = await generator.generate_task(
                patterns=patterns,
                task_type="feature",
                seed_prompts=["Add logging"],
                provider="nim"
            )

            assert task.prompt == "Add caching to the database queries"
            assert task.task_type == "feature"
            assert task.repo == "ghostwork"
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_task_with_anthropic(self):
        """Should call Anthropic to generate a synthetic task."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"bugfix": 1.0},
            repo_name="test-repo",
            common_paths=["lib/"],
            languages={"typescript": 1.0}
        )

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Fix memory leak in worker thread"

            task = await generator.generate_task(
                patterns=patterns,
                task_type="bugfix",
                seed_prompts=["Fix timeout error"],
                provider="anthropic"
            )

            assert task.prompt == "Fix memory leak in worker thread"
            assert task.task_type == "bugfix"
            assert task.repo == "test-repo"
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_routes_to_nim(self):
        """_call_llm should route to NIM when provider is 'nim'."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        with patch.object(generator, '_call_nim', new_callable=AsyncMock) as mock_nim:
            mock_nim.return_value = "Generated task"

            result = await generator._call_llm("Test prompt", provider="nim")

            assert result == "Generated task"
            mock_nim.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_call_llm_routes_to_anthropic(self):
        """_call_llm should route to Anthropic when provider is 'anthropic'."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        with patch.object(generator, '_call_anthropic', new_callable=AsyncMock) as mock_anthropic:
            mock_anthropic.return_value = "Generated task from Claude"

            result = await generator._call_llm("Test prompt", provider="anthropic")

            assert result == "Generated task from Claude"
            mock_anthropic.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_call_llm_unknown_provider_raises(self):
        """_call_llm should raise ValueError for unknown provider."""
        generator = SyntheticGenerator()

        with pytest.raises(ValueError, match="Unknown provider"):
            await generator._call_llm("Test prompt", provider="unknown")

    @pytest.mark.asyncio
    async def test_call_nim_makes_http_request(self):
        """_call_nim should make HTTP request to NIM endpoint."""
        from unittest.mock import AsyncMock, patch, MagicMock
        import httpx

        generator = SyntheticGenerator()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  Generated response  "}}]
        }

        with patch.dict('os.environ', {
            'NVIDIA_API_KEY': 'test-key',
            'NIM_ENDPOINT': 'https://test.nvidia.com/v1',
            'NIM_MODEL': 'test-model'
        }):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await generator._call_nim("Test prompt")

                assert result == "Generated response"
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert "test.nvidia.com" in call_args[0][0]
                assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_call_anthropic_makes_api_request(self):
        """_call_anthropic should call Anthropic API."""
        from unittest.mock import AsyncMock, patch, MagicMock

        generator = SyntheticGenerator()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  Claude response  ")]

        with patch('anthropic.AsyncAnthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic_class.return_value = mock_client

            result = await generator._call_anthropic("Test prompt")

            assert result == "Claude response"
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_task_returns_synth_task_id(self):
        """generate_task should return SyntheticTask with synth_ prefixed task_id."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"refactor": 1.0},
            repo_name="myrepo",
            common_paths=["src/"],
            languages={"python": 1.0}
        )

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Refactor the module"

            task = await generator.generate_task(
                patterns=patterns,
                task_type="refactor",
                seed_prompts=["Clean up code"],
                provider="nim"
            )

            assert task.task_id.startswith("synth_")
            assert len(task.task_id) == 14  # "synth_" + 8 hex chars
            assert task.target_files == []
            assert task.expected_tools == ["Read", "Edit"]
            assert task.source_pattern_id == ""

    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Should generate multiple tasks with progress callback."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"feature": 0.5, "bugfix": 0.5},
            repo_name="ghostwork",
            languages={"python": 1.0}
        )

        progress_updates = []

        def on_progress(current: int, total: int):
            progress_updates.append((current, total))

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated task"

            tasks = await generator.generate_batch(
                patterns=patterns,
                seed_prompts=["Add feature", "Fix bug"],
                count=5,
                provider="nim",
                on_progress=on_progress
            )

            assert len(tasks) == 5
            assert len(progress_updates) == 5
            assert progress_updates[-1] == (5, 5)

    @pytest.mark.asyncio
    async def test_generate_batch_samples_task_types(self):
        """Should sample task types from the patterns distribution."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        # 100% bugfix to verify task type is passed correctly
        patterns = TracePatterns(
            task_types={"bugfix": 1.0},
            repo_name="testproject",
            languages={"python": 1.0}
        )

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Fix the bug"

            tasks = await generator.generate_batch(
                patterns=patterns,
                seed_prompts=["Fix issue"],
                count=3,
                provider="nim"
            )

            # All tasks should be bugfix type since distribution is 100% bugfix
            assert all(task.task_type == "bugfix" for task in tasks)

    @pytest.mark.asyncio
    async def test_generate_batch_handles_errors_gracefully(self):
        """Should continue generating even if some tasks fail."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"feature": 1.0},
            repo_name="ghostwork",
            languages={"python": 1.0}
        )

        call_count = 0

        async def mock_call_llm(prompt, provider="nim"):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("LLM error")
            return "Generated task"

        with patch.object(generator, '_call_llm', side_effect=mock_call_llm):
            tasks = await generator.generate_batch(
                patterns=patterns,
                seed_prompts=["Add feature"],
                count=3,
                provider="nim"
            )

            # Should have 2 successful tasks (first and third)
            assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_generate_batch_defaults_to_feature(self):
        """Should default to 'feature' task type if no task_types in patterns."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        # Empty task_types
        patterns = TracePatterns(
            task_types={},
            repo_name="ghostwork",
            languages={"python": 1.0}
        )

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated task"

            tasks = await generator.generate_batch(
                patterns=patterns,
                seed_prompts=["Do something"],
                count=2,
                provider="nim"
            )

            # All tasks should default to "feature"
            assert all(task.task_type == "feature" for task in tasks)

    @pytest.mark.asyncio
    async def test_generate_batch_no_progress_callback(self):
        """Should work without a progress callback."""
        from unittest.mock import AsyncMock, patch
        from bashgym.factory.pattern_extractor import TracePatterns

        generator = SyntheticGenerator()

        patterns = TracePatterns(
            task_types={"feature": 1.0},
            repo_name="ghostwork",
            languages={"python": 1.0}
        )

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated task"

            # Should not raise even without on_progress
            tasks = await generator.generate_batch(
                patterns=patterns,
                seed_prompts=["Add feature"],
                count=3,
                provider="nim"
            )

            assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_schema_driven_strategy(self):
        """Should generate tasks from repo schema."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        repo_schema = {
            "name": "ghostwork",
            "structure": {
                "src/": ["api.py", "auth.py", "utils.py"],
                "tests/": ["test_api.py", "test_auth.py"],
            },
            "frameworks": ["fastapi", "pytest"]
        }

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Add rate limiting to the API endpoints"

            tasks = await generator.generate_from_schema(
                repo_schema=repo_schema,
                count=5,
                provider="nim"
            )

            assert len(tasks) == 5
            assert all(t.repo == "ghostwork" for t in tasks)
            assert all(t.task_id.startswith("schema_") for t in tasks)
            assert all(t.metadata.get("strategy") == "schema_driven" for t in tasks)
            mock_llm.assert_called()

    @pytest.mark.asyncio
    async def test_augmented_strategy(self):
        """Should generate variations using augmented strategy."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        seed_examples = [
            {"prompt": "Add logging to API", "response": "I'll add logging..."},
            {"prompt": "Fix auth bug", "response": "I'll fix the bug..."},
        ]

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Add comprehensive logging to the API client"

            tasks = await generator.generate_augmented(
                seed_examples=seed_examples,
                variations_per_seed=2,
                provider="nim"
            )

            assert len(tasks) == 4  # 2 seeds * 2 variations
            assert all(task.task_id.startswith("aug_") for task in tasks)
            assert all(task.metadata.get("strategy") == "augmented" for task in tasks)
            assert mock_llm.call_count == 4

    @pytest.mark.asyncio
    async def test_augmented_strategy_stores_seed_prompt(self):
        """Augmented tasks should store original seed prompt in metadata."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        seed_examples = [
            {"prompt": "Implement caching", "id": "seed_001", "repo": "myrepo"},
        ]

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Add Redis caching layer"

            tasks = await generator.generate_augmented(
                seed_examples=seed_examples,
                variations_per_seed=1,
                provider="nim"
            )

            assert len(tasks) == 1
            assert tasks[0].metadata["seed_prompt"] == "Implement caching"
            assert tasks[0].source_pattern_id == "seed_001"
            assert tasks[0].repo == "myrepo"

    @pytest.mark.asyncio
    async def test_augmented_strategy_handles_errors(self):
        """Augmented strategy should skip failed variations and continue."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        seed_examples = [
            {"prompt": "Add logging"},
        ]

        call_count = 0

        async def mock_call_llm(prompt, provider="nim"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM error")
            return "Generated variation"

        with patch.object(generator, '_call_llm', side_effect=mock_call_llm):
            tasks = await generator.generate_augmented(
                seed_examples=seed_examples,
                variations_per_seed=3,
                provider="nim"
            )

            # First call fails, next 2 succeed
            assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_augmented_strategy_default_variations(self):
        """Augmented strategy should default to 3 variations per seed."""
        from unittest.mock import AsyncMock, patch

        generator = SyntheticGenerator()

        seed_examples = [
            {"prompt": "Test prompt"},
        ]

        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Variation"

            tasks = await generator.generate_augmented(
                seed_examples=seed_examples,
                provider="nim"
            )

            # Default is 3 variations per seed
            assert len(tasks) == 3
            assert mock_llm.call_count == 3


class TestExportToNemo:
    """Tests for NeMo JSONL export functionality."""

    def test_export_to_nemo(self):
        """Should export synthetic tasks to NeMo JSONL format."""
        import json
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        tasks = [
            SyntheticTask(
                task_id="synth_001",
                prompt="Add logging to API",
                target_files=["src/api.py"],
                task_type="feature",
                expected_tools=["Read", "Edit"],
                source_pattern_id="p1",
                repo="ghostwork"
            ),
            SyntheticTask(
                task_id="synth_002",
                prompt="Fix auth bug",
                target_files=["src/auth.py"],
                task_type="bugfix",
                expected_tools=["Read", "Edit", "Bash"],
                source_pattern_id="p2",
                repo="ghostwork"
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            result = generator.export_to_nemo(
                tasks=tasks,
                output_dir=output_path,
                train_ratio=0.5  # 50% train for this test
            )

            assert (output_path / "train.jsonl").exists()
            assert (output_path / "val.jsonl").exists()
            assert (output_path / "metadata.json").exists()

            # Check JSONL format
            with open(output_path / "train.jsonl") as f:
                line = f.readline()
                data = json.loads(line)
                assert "messages" in data
                assert data["messages"][0]["role"] == "system"
                assert data["messages"][1]["role"] == "user"

    def test_export_to_nemo_creates_output_dir(self):
        """Should create output directory if it doesn't exist."""
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        tasks = [
            SyntheticTask(
                task_id="synth_001",
                prompt="Add feature",
                target_files=["src/main.py"],
                task_type="feature",
                expected_tools=["Read", "Edit"],
                source_pattern_id="p1",
                repo="test-repo"
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested directory that doesn't exist
            output_path = Path(tmpdir) / "deep" / "nested" / "output"

            generator.export_to_nemo(tasks=tasks, output_dir=output_path)

            assert output_path.exists()
            assert (output_path / "train.jsonl").exists()

    def test_export_to_nemo_metadata(self):
        """Should write correct metadata.json with generation info."""
        import json
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        tasks = [
            SyntheticTask(
                task_id=f"synth_{i:03d}",
                prompt=f"Task {i}",
                target_files=["src/file.py"],
                task_type="feature",
                expected_tools=["Read", "Edit"],
                source_pattern_id="p1",
                repo="test-repo"
            )
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            result = generator.export_to_nemo(
                tasks=tasks,
                output_dir=output_path,
                train_ratio=0.8
            )

            # Check returned metadata
            assert result["total_examples"] == 10
            assert result["train_examples"] == 8
            assert result["val_examples"] == 2
            assert result["train_ratio"] == 0.8
            assert "generated_at" in result
            assert result["strategy"] == "trace_seeded"

            # Check written metadata.json
            with open(output_path / "metadata.json") as f:
                written_metadata = json.load(f)
                assert written_metadata["total_examples"] == 10

    def test_export_to_nemo_task_metadata_in_jsonl(self):
        """Should include task metadata in each JSONL entry."""
        import json
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        tasks = [
            SyntheticTask(
                task_id="synth_001",
                prompt="Add caching",
                target_files=["src/cache.py"],
                task_type="feature",
                expected_tools=["Read", "Edit"],
                source_pattern_id="pattern_123",
                repo="myrepo"
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            generator.export_to_nemo(
                tasks=tasks,
                output_dir=output_path,
                train_ratio=1.0  # All to train
            )

            with open(output_path / "train.jsonl") as f:
                data = json.loads(f.readline())
                assert "metadata" in data
                assert data["metadata"]["synthetic"] is True
                assert data["metadata"]["task_type"] == "feature"
                assert data["metadata"]["repo"] == "myrepo"
                assert data["metadata"]["task_id"] == "synth_001"

    def test_export_to_nemo_custom_system_prompt(self):
        """Should use custom system prompt when provided."""
        import json
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        tasks = [
            SyntheticTask(
                task_id="synth_001",
                prompt="Fix bug",
                target_files=["src/bug.py"],
                task_type="bugfix",
                expected_tools=["Read", "Edit"],
                source_pattern_id="p1",
                repo="test-repo"
            ),
        ]

        custom_system = "You are an expert Python developer specializing in debugging."

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            generator.export_to_nemo(
                tasks=tasks,
                output_dir=output_path,
                train_ratio=1.0,  # Ensure task goes to train.jsonl
                system_prompt=custom_system
            )

            with open(output_path / "train.jsonl") as f:
                data = json.loads(f.readline())
                assert data["messages"][0]["content"] == custom_system

    def test_export_to_nemo_empty_tasks_list(self):
        """Should handle empty tasks list gracefully."""
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            result = generator.export_to_nemo(
                tasks=[],
                output_dir=output_path
            )

            assert result["total_examples"] == 0
            assert result["train_examples"] == 0
            assert result["val_examples"] == 0

            # Files should still be created (empty)
            assert (output_path / "train.jsonl").exists()
            assert (output_path / "val.jsonl").exists()

    def test_export_to_nemo_default_train_ratio(self):
        """Should use 0.9 as default train ratio."""
        import tempfile
        from pathlib import Path

        generator = SyntheticGenerator()

        tasks = [
            SyntheticTask(
                task_id=f"synth_{i:03d}",
                prompt=f"Task {i}",
                target_files=["src/file.py"],
                task_type="feature",
                expected_tools=["Read"],
                source_pattern_id="p1",
                repo="test-repo"
            )
            for i in range(100)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"

            result = generator.export_to_nemo(
                tasks=tasks,
                output_dir=output_path
            )

            # Default 0.9 ratio: 90 train, 10 val
            assert result["train_examples"] == 90
            assert result["val_examples"] == 10