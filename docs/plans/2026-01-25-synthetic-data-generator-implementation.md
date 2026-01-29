# Synthetic Data Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement synthetic data generation from 84+ seed traces to create 500-2000+ training examples for LoRA fine-tuning.

**Architecture:** Pattern extraction from traces → LLM-powered task generation → NeMo-compatible JSONL output. Three strategies: trace_seeded (primary), augmented, schema_driven.

**Tech Stack:** Python 3.12, FastAPI, NVIDIA NIM (qwen2.5-coder-32b), React/TypeScript frontend

---

## Task 1: Create TracePatterns Data Structures

**Files:**
- Create: `bashgym/factory/pattern_extractor.py`
- Test: `tests/factory/test_pattern_extractor.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_pattern_extractor.py
import pytest
from bashgym.factory.pattern_extractor import TracePatterns, FileCluster, ToolSequence

def test_trace_patterns_dataclass():
    """TracePatterns should store extracted patterns from traces."""
    patterns = TracePatterns(
        task_types={"feature": 0.4, "bugfix": 0.3, "refactor": 0.2, "spec": 0.1},
        file_clusters=[FileCluster(patterns=["src/*.py", "tests/*.py"], frequency=0.6)],
        common_paths=["src/", "bashgym/"],
        languages={"python": 0.7, "typescript": 0.3},
        tool_patterns=[ToolSequence(tools=["Read", "Edit", "Bash"], frequency=0.5)],
        avg_tool_calls=12,
        prompt_templates=["Fix the {issue} in {file}"],
        prompt_keywords=["implement", "fix", "add"],
        repo_name="bashgym",
        framework_hints=["fastapi", "pytest"]
    )

    assert patterns.task_types["feature"] == 0.4
    assert len(patterns.file_clusters) == 1
    assert patterns.repo_name == "bashgym"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_pattern_extractor.py::test_trace_patterns_dataclass -v`
Expected: FAIL with "No module named 'bashgym.factory.pattern_extractor'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/pattern_extractor.py
"""Pattern extraction from execution traces for synthetic data generation."""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class FileCluster:
    """Files that are frequently modified together."""
    patterns: List[str]  # Glob patterns like ["src/*.py", "tests/*.py"]
    frequency: float     # How often this cluster appears (0-1)

@dataclass
class ToolSequence:
    """Common tool call sequences."""
    tools: List[str]     # ["Read", "Edit", "Bash"]
    frequency: float     # How often this sequence appears (0-1)

@dataclass
class TracePatterns:
    """Extracted patterns from a set of traces."""
    # Task classification
    task_types: Dict[str, float] = field(default_factory=dict)

    # File patterns
    file_clusters: List[FileCluster] = field(default_factory=list)
    common_paths: List[str] = field(default_factory=list)
    languages: Dict[str, float] = field(default_factory=dict)

    # Tool sequences
    tool_patterns: List[ToolSequence] = field(default_factory=list)
    avg_tool_calls: int = 0

    # Prompt patterns
    prompt_templates: List[str] = field(default_factory=list)
    prompt_keywords: List[str] = field(default_factory=list)

    # Repo context
    repo_name: str = ""
    framework_hints: List[str] = field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_pattern_extractor.py::test_trace_patterns_dataclass -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/factory/test_pattern_extractor.py bashgym/factory/pattern_extractor.py
git commit -m "feat(factory): add TracePatterns data structures for pattern extraction"
```

---

## Task 2: Implement Task Type Classification

**Files:**
- Modify: `bashgym/factory/pattern_extractor.py`
- Test: `tests/factory/test_pattern_extractor.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_pattern_extractor.py (add to existing)
def test_classify_task_type_feature():
    """Should classify 'add' and 'implement' prompts as feature."""
    extractor = PatternExtractor()

    assert extractor._classify_task_type("Add a new login button") == "feature"
    assert extractor._classify_task_type("Implement user authentication") == "feature"
    assert extractor._classify_task_type("Create the API endpoint") == "feature"

def test_classify_task_type_bugfix():
    """Should classify 'fix' and 'bug' prompts as bugfix."""
    extractor = PatternExtractor()

    assert extractor._classify_task_type("Fix the login bug") == "bugfix"
    assert extractor._classify_task_type("Debug the error in utils") == "bugfix"
    assert extractor._classify_task_type("Resolve the crash on startup") == "bugfix"

def test_classify_task_type_refactor():
    """Should classify refactoring prompts correctly."""
    extractor = PatternExtractor()

    assert extractor._classify_task_type("Refactor the database module") == "refactor"
    assert extractor._classify_task_type("Clean up the utils code") == "refactor"
    assert extractor._classify_task_type("Improve the API structure") == "refactor"

def test_classify_task_type_spec():
    """Should classify spec/design prompts correctly."""
    extractor = PatternExtractor()

    assert extractor._classify_task_type("Write a spec for the new feature") == "spec"
    assert extractor._classify_task_type("Design the authentication flow") == "spec"
    assert extractor._classify_task_type("Plan the migration strategy") == "spec"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_pattern_extractor.py::test_classify_task_type_feature -v`
Expected: FAIL with "PatternExtractor has no attribute '_classify_task_type'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/pattern_extractor.py (add to existing)
import re

class PatternExtractor:
    """Extracts patterns from execution traces."""

    # Keywords for task classification
    TASK_KEYWORDS = {
        "feature": ["add", "implement", "create", "build", "new", "introduce"],
        "bugfix": ["fix", "bug", "error", "debug", "resolve", "crash", "issue", "broken"],
        "refactor": ["refactor", "clean", "improve", "reorganize", "restructure", "simplify"],
        "spec": ["spec", "design", "plan", "architect", "document", "define", "outline"]
    }

    def _classify_task_type(self, prompt: str) -> str:
        """Classify a prompt into task type based on keywords."""
        prompt_lower = prompt.lower()

        # Count keyword matches for each type
        scores = {}
        for task_type, keywords in self.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return "feature"  # Default to feature

        # Return type with highest score
        return max(scores, key=scores.get)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_pattern_extractor.py -k "classify_task_type" -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add bashgym/factory/pattern_extractor.py tests/factory/test_pattern_extractor.py
git commit -m "feat(factory): add task type classification to PatternExtractor"
```

---

## Task 3: Implement Language Detection from File Paths

**Files:**
- Modify: `bashgym/factory/pattern_extractor.py`
- Test: `tests/factory/test_pattern_extractor.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_pattern_extractor.py (add to existing)
def test_detect_languages():
    """Should detect languages from file paths."""
    extractor = PatternExtractor()

    files = [
        "src/main.py",
        "src/utils.py",
        "tests/test_main.py",
        "frontend/App.tsx",
        "README.md"
    ]

    languages = extractor._detect_languages(files)

    assert languages["python"] == pytest.approx(0.6, rel=0.1)
    assert languages["typescript"] == pytest.approx(0.2, rel=0.1)
    assert languages["markdown"] == pytest.approx(0.2, rel=0.1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_pattern_extractor.py::test_detect_languages -v`
Expected: FAIL with "PatternExtractor has no attribute '_detect_languages'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/pattern_extractor.py (add to PatternExtractor class)
    # File extension to language mapping
    EXTENSION_LANGUAGES = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "bash",
        ".rs": "rust",
        ".go": "go",
    }

    def _detect_languages(self, file_paths: List[str]) -> Dict[str, float]:
        """Detect language distribution from file paths."""
        from pathlib import Path

        if not file_paths:
            return {}

        lang_counts: Dict[str, int] = {}

        for path in file_paths:
            ext = Path(path).suffix.lower()
            lang = self.EXTENSION_LANGUAGES.get(ext, "other")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Convert to frequencies
        total = sum(lang_counts.values())
        return {lang: count / total for lang, count in lang_counts.items()}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_pattern_extractor.py::test_detect_languages -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/pattern_extractor.py tests/factory/test_pattern_extractor.py
git commit -m "feat(factory): add language detection from file paths"
```

---

## Task 4: Implement Tool Sequence Extraction (N-gram Mining)

**Files:**
- Modify: `bashgym/factory/pattern_extractor.py`
- Test: `tests/factory/test_pattern_extractor.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_pattern_extractor.py (add to existing)
def test_extract_tool_sequences():
    """Should extract common tool call sequences."""
    extractor = PatternExtractor()

    # Simulate tool sequences from multiple traces
    tool_sequences = [
        ["Read", "Edit", "Bash"],
        ["Read", "Edit", "Bash"],
        ["Glob", "Read", "Edit"],
        ["Read", "Edit", "Bash"],
        ["Read", "Write", "Bash"],
    ]

    patterns = extractor._extract_tool_sequences(tool_sequences, n=3, min_frequency=0.3)

    # "Read, Edit, Bash" appears 3/5 times = 0.6
    assert any(p.tools == ["Read", "Edit", "Bash"] for p in patterns)
    assert patterns[0].frequency >= 0.3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_pattern_extractor.py::test_extract_tool_sequences -v`
Expected: FAIL with "PatternExtractor has no attribute '_extract_tool_sequences'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/pattern_extractor.py (add to PatternExtractor class)
from collections import Counter

    def _extract_tool_sequences(
        self,
        tool_lists: List[List[str]],
        n: int = 3,
        min_frequency: float = 0.2
    ) -> List[ToolSequence]:
        """Extract common N-gram tool sequences."""
        if not tool_lists:
            return []

        # Extract N-grams from each sequence
        ngram_counts: Counter = Counter()

        for tools in tool_lists:
            for i in range(len(tools) - n + 1):
                ngram = tuple(tools[i:i + n])
                ngram_counts[ngram] += 1

        # Convert to frequencies
        total_sequences = len(tool_lists)
        patterns = []

        for ngram, count in ngram_counts.most_common():
            frequency = count / total_sequences
            if frequency >= min_frequency:
                patterns.append(ToolSequence(
                    tools=list(ngram),
                    frequency=frequency
                ))

        return patterns
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_pattern_extractor.py::test_extract_tool_sequences -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/pattern_extractor.py tests/factory/test_pattern_extractor.py
git commit -m "feat(factory): add tool sequence N-gram extraction"
```

---

## Task 5: Implement Full Pattern Extraction from Traces

**Files:**
- Modify: `bashgym/factory/pattern_extractor.py`
- Test: `tests/factory/test_pattern_extractor.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_pattern_extractor.py (add to existing)
def test_extract_patterns_from_traces():
    """Should extract patterns from a list of trace dicts."""
    extractor = PatternExtractor()

    # Mock trace data (simplified structure matching actual traces)
    traces = [
        {
            "metadata": {"repo": "bashgym"},
            "summary": {"task_description": "Add retry logic to API client"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/api.py"}},
                {"tool": "Edit", "input": {"file_path": "src/api.py"}},
                {"tool": "Bash", "input": {"command": "pytest tests/"}},
            ]
        },
        {
            "metadata": {"repo": "bashgym"},
            "summary": {"task_description": "Fix bug in authentication"},
            "trace": [
                {"tool": "Glob", "input": {"pattern": "src/*.py"}},
                {"tool": "Read", "input": {"file_path": "src/auth.py"}},
                {"tool": "Edit", "input": {"file_path": "src/auth.py"}},
            ]
        },
    ]

    patterns = extractor.extract_patterns(traces, repo_name="bashgym")

    assert patterns.repo_name == "bashgym"
    assert "feature" in patterns.task_types or "bugfix" in patterns.task_types
    assert patterns.avg_tool_calls == 3
    assert "python" in patterns.languages
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_pattern_extractor.py::test_extract_patterns_from_traces -v`
Expected: FAIL with "PatternExtractor has no attribute 'extract_patterns'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/pattern_extractor.py (add to PatternExtractor class)
    def extract_patterns(self, traces: List[Dict], repo_name: str = "") -> TracePatterns:
        """Extract patterns from a list of trace dictionaries."""
        if not traces:
            return TracePatterns(repo_name=repo_name)

        # Collect data from all traces
        task_type_counts: Dict[str, int] = {}
        all_files: List[str] = []
        all_tool_sequences: List[List[str]] = []
        all_prompts: List[str] = []
        total_tool_calls = 0

        for trace in traces:
            # Extract prompt/task description
            summary = trace.get("summary", {})
            prompt = summary.get("task_description", "") or summary.get("prompt", "")
            if prompt:
                all_prompts.append(prompt)
                task_type = self._classify_task_type(prompt)
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

            # Extract tool sequence and files
            trace_steps = trace.get("trace", [])
            tools = []
            for step in trace_steps:
                tool_name = step.get("tool", "")
                if tool_name:
                    tools.append(tool_name)

                # Extract file paths from tool inputs
                input_data = step.get("input", {})
                if "file_path" in input_data:
                    all_files.append(input_data["file_path"])
                if "path" in input_data:
                    all_files.append(input_data["path"])

            if tools:
                all_tool_sequences.append(tools)
                total_tool_calls += len(tools)

        # Calculate distributions
        total_traces = len(traces)
        task_types = {t: c / total_traces for t, c in task_type_counts.items()}
        languages = self._detect_languages(all_files)
        tool_patterns = self._extract_tool_sequences(all_tool_sequences)
        avg_tool_calls = total_tool_calls // total_traces if total_traces > 0 else 0

        # Extract common paths
        common_paths = self._extract_common_paths(all_files)

        # Extract keywords from prompts
        prompt_keywords = self._extract_keywords(all_prompts)

        return TracePatterns(
            task_types=task_types,
            file_clusters=[],  # TODO: implement clustering
            common_paths=common_paths,
            languages=languages,
            tool_patterns=tool_patterns,
            avg_tool_calls=avg_tool_calls,
            prompt_templates=[],  # TODO: implement template extraction
            prompt_keywords=prompt_keywords,
            repo_name=repo_name,
            framework_hints=[]  # TODO: detect frameworks
        )

    def _extract_common_paths(self, file_paths: List[str]) -> List[str]:
        """Extract common directory prefixes."""
        from pathlib import Path
        from collections import Counter

        if not file_paths:
            return []

        dir_counts: Counter = Counter()
        for path in file_paths:
            parts = Path(path).parts
            for i in range(1, len(parts)):
                prefix = "/".join(parts[:i]) + "/"
                dir_counts[prefix] += 1

        # Return directories that appear in >20% of files
        threshold = len(file_paths) * 0.2
        return [d for d, c in dir_counts.most_common(10) if c >= threshold]

    def _extract_keywords(self, prompts: List[str]) -> List[str]:
        """Extract common action keywords from prompts."""
        all_keywords = []
        for task_type, keywords in self.TASK_KEYWORDS.items():
            all_keywords.extend(keywords)

        found = set()
        for prompt in prompts:
            prompt_lower = prompt.lower()
            for kw in all_keywords:
                if kw in prompt_lower:
                    found.add(kw)

        return list(found)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_pattern_extractor.py::test_extract_patterns_from_traces -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/pattern_extractor.py tests/factory/test_pattern_extractor.py
git commit -m "feat(factory): add full pattern extraction from traces"
```

---

## Task 6: Create SyntheticTask Data Structures

**Files:**
- Create: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py
import pytest
from bashgym.factory.synthetic_generator import SyntheticTask, GenerationPreset, PRESETS

def test_synthetic_task_dataclass():
    """SyntheticTask should store generated task data."""
    task = SyntheticTask(
        task_id="synth_001",
        prompt="Add logging to the API client",
        target_files=["src/api.py", "tests/test_api.py"],
        task_type="feature",
        expected_tools=["Read", "Edit", "Bash"],
        source_pattern_id="pattern_003",
        repo="bashgym"
    )

    assert task.task_id == "synth_001"
    assert task.task_type == "feature"
    assert len(task.target_files) == 2

def test_presets_exist():
    """Should have all required presets defined."""
    assert "quick_test" in PRESETS
    assert "balanced" in PRESETS
    assert "production" in PRESETS
    assert "custom" in PRESETS

    assert PRESETS["quick_test"].target_examples == 100
    assert PRESETS["balanced"].target_examples == 500
    assert PRESETS["production"].target_examples == 2000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py -v`
Expected: FAIL with "No module named 'bashgym.factory.synthetic_generator'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py
"""Synthetic task generation from extracted patterns."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

@dataclass
class SyntheticTask:
    """A synthetically generated task."""
    task_id: str
    prompt: str
    target_files: List[str]
    task_type: str
    expected_tools: List[str]
    source_pattern_id: str
    repo: str
    metadata: Dict = field(default_factory=dict)

class GenerationStrategy(str, Enum):
    TRACE_SEEDED = "trace_seeded"
    AUGMENTED = "augmented"
    SCHEMA_DRIVEN = "schema_driven"

@dataclass
class GenerationPreset:
    """Preset configuration for generation."""
    label: str
    description: str
    target_examples: Optional[int]
    multiplier: Optional[int] = None  # None means auto-calculate

# Research-based presets
PRESETS: Dict[str, GenerationPreset] = {
    "quick_test": GenerationPreset(
        label="Quick Test",
        description="Fast iteration, minimal generation",
        target_examples=100,
    ),
    "balanced": GenerationPreset(
        label="Balanced (Recommended)",
        description="Good quality/time tradeoff for LoRA",
        target_examples=500,
    ),
    "production": GenerationPreset(
        label="Production",
        description="Full dataset for best results",
        target_examples=2000,
    ),
    "custom": GenerationPreset(
        label="Custom",
        description="Set your own target",
        target_examples=None,
    ),
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add SyntheticTask and GenerationPreset data structures"
```

---

## Task 7: Implement Auto-Multiplier Calculation

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
def test_calculate_multiplier():
    """Should calculate multiplier from seed count and target."""
    generator = SyntheticGenerator()

    # 84 seeds, want 500 → ceil(500/84) = 6
    assert generator.calculate_multiplier(seed_count=84, target_examples=500) == 6

    # 84 seeds, want 100 → ceil(100/84) = 2
    assert generator.calculate_multiplier(seed_count=84, target_examples=100) == 2

    # 84 seeds, want 2000 → ceil(2000/84) = 24
    assert generator.calculate_multiplier(seed_count=84, target_examples=2000) == 24

def test_calculate_multiplier_edge_cases():
    """Should handle edge cases."""
    generator = SyntheticGenerator()

    # Minimum multiplier of 1
    assert generator.calculate_multiplier(seed_count=1000, target_examples=100) == 1

    # Zero seeds returns 0
    assert generator.calculate_multiplier(seed_count=0, target_examples=500) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_calculate_multiplier -v`
Expected: FAIL with "SyntheticGenerator has no attribute 'calculate_multiplier'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add class)
import math

class SyntheticGenerator:
    """Generates synthetic tasks from extracted patterns."""

    def __init__(self):
        self.presets = PRESETS

    def calculate_multiplier(self, seed_count: int, target_examples: int) -> int:
        """Calculate multiplier needed to reach target examples."""
        if seed_count == 0:
            return 0

        multiplier = math.ceil(target_examples / seed_count)
        return max(1, multiplier)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py -k "multiplier" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add auto-multiplier calculation"
```

---

## Task 8: Implement Task Generation Prompt Building

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
from bashgym.factory.pattern_extractor import TracePatterns

def test_build_generation_prompt():
    """Should build LLM prompt for task generation."""
    generator = SyntheticGenerator()

    patterns = TracePatterns(
        task_types={"feature": 0.5, "bugfix": 0.5},
        repo_name="bashgym",
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

    assert "bashgym" in prompt
    assert "feature" in prompt
    assert "fastapi" in prompt
    assert "Add retry logic" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_build_generation_prompt -v`
Expected: FAIL with "SyntheticGenerator has no attribute '_build_generation_prompt'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add to SyntheticGenerator class)
from bashgym.factory.pattern_extractor import TracePatterns

    def _build_generation_prompt(
        self,
        patterns: TracePatterns,
        task_type: str,
        seed_prompts: List[str],
        target_files: Optional[List[str]] = None
    ) -> str:
        """Build prompt for LLM to generate synthetic tasks."""
        frameworks = ", ".join(patterns.framework_hints) if patterns.framework_hints else "general"
        paths = ", ".join(patterns.common_paths[:5]) if patterns.common_paths else "various"

        examples = "\n".join(f'- "{p}"' for p in seed_prompts[:5])

        prompt = f"""You are generating coding tasks for the {patterns.repo_name} codebase.

Task type: {task_type}
Common directories: {paths}
Frameworks: {frameworks}
Primary language: {max(patterns.languages, key=patterns.languages.get) if patterns.languages else "python"}

Example real tasks from this repo:
{examples}

Generate a new, realistic {task_type} task that:
- Matches the style and complexity of the examples
- Is specific and actionable
- Would require modifying 1-3 files
- Uses realistic file names for this project

Return ONLY the task prompt, nothing else. No quotes, no explanation."""

        return prompt
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py::test_build_generation_prompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add task generation prompt building"
```

---

## Task 9: Implement NIM Client Integration

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_generate_task_with_nim():
    """Should call NIM to generate a synthetic task."""
    generator = SyntheticGenerator()

    patterns = TracePatterns(
        task_types={"feature": 1.0},
        repo_name="bashgym",
        common_paths=["src/"],
        languages={"python": 1.0}
    )

    # Mock the NIM client
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
        assert task.repo == "bashgym"
        mock_llm.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_generate_task_with_nim -v`
Expected: FAIL with "SyntheticGenerator has no attribute 'generate_task'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add to SyntheticGenerator class)
import httpx
import os
import uuid

    async def _call_llm(self, prompt: str, provider: str = "nim") -> str:
        """Call LLM provider to generate text."""
        if provider == "nim":
            return await self._call_nim(prompt)
        elif provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _call_nim(self, prompt: str) -> str:
        """Call NVIDIA NIM API."""
        endpoint = os.getenv("NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
        model = os.getenv("NIM_MODEL", "qwen/qwen2.5-coder-32b-instruct")
        api_key = os.getenv("NVIDIA_API_KEY", "")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.8
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        import anthropic

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    async def generate_task(
        self,
        patterns: TracePatterns,
        task_type: str,
        seed_prompts: List[str],
        provider: str = "nim"
    ) -> SyntheticTask:
        """Generate a single synthetic task."""
        prompt = self._build_generation_prompt(patterns, task_type, seed_prompts)

        generated_prompt = await self._call_llm(prompt, provider)

        return SyntheticTask(
            task_id=f"synth_{uuid.uuid4().hex[:8]}",
            prompt=generated_prompt,
            target_files=[],  # Could be inferred from patterns
            task_type=task_type,
            expected_tools=["Read", "Edit"],  # Default tools
            source_pattern_id="",
            repo=patterns.repo_name
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py::test_generate_task_with_nim -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add NIM client integration for task generation"
```

---

## Task 10: Implement Batch Generation with Progress

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
from dataclasses import dataclass

@pytest.mark.asyncio
async def test_generate_batch():
    """Should generate multiple tasks with progress callback."""
    generator = SyntheticGenerator()

    patterns = TracePatterns(
        task_types={"feature": 0.5, "bugfix": 0.5},
        repo_name="bashgym",
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_generate_batch -v`
Expected: FAIL with "SyntheticGenerator has no attribute 'generate_batch'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add to SyntheticGenerator class)
import random
from typing import Callable

    async def generate_batch(
        self,
        patterns: TracePatterns,
        seed_prompts: List[str],
        count: int,
        provider: str = "nim",
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[SyntheticTask]:
        """Generate a batch of synthetic tasks."""
        tasks = []
        task_types = list(patterns.task_types.keys()) if patterns.task_types else ["feature"]
        weights = list(patterns.task_types.values()) if patterns.task_types else [1.0]

        for i in range(count):
            # Sample task type based on distribution
            task_type = random.choices(task_types, weights=weights, k=1)[0]

            try:
                task = await self.generate_task(
                    patterns=patterns,
                    task_type=task_type,
                    seed_prompts=seed_prompts,
                    provider=provider
                )
                tasks.append(task)
            except Exception as e:
                # Log error but continue
                print(f"Failed to generate task {i}: {e}")

            if on_progress:
                on_progress(i + 1, count)

        return tasks
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py::test_generate_batch -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add batch generation with progress tracking"
```

---

## Task 11: Implement NeMo JSONL Export

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
import json
import tempfile
from pathlib import Path

def test_export_to_nemo():
    """Should export synthetic tasks to NeMo JSONL format."""
    generator = SyntheticGenerator()

    tasks = [
        SyntheticTask(
            task_id="synth_001",
            prompt="Add logging to API",
            target_files=["src/api.py"],
            task_type="feature",
            expected_tools=["Read", "Edit"],
            source_pattern_id="p1",
            repo="bashgym"
        ),
        SyntheticTask(
            task_id="synth_002",
            prompt="Fix auth bug",
            target_files=["src/auth.py"],
            task_type="bugfix",
            expected_tools=["Read", "Edit", "Bash"],
            source_pattern_id="p2",
            repo="bashgym"
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_export_to_nemo -v`
Expected: FAIL with "SyntheticGenerator has no attribute 'export_to_nemo'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add to SyntheticGenerator class)
import json
from pathlib import Path
from datetime import datetime

    def export_to_nemo(
        self,
        tasks: List[SyntheticTask],
        output_dir: Path,
        train_ratio: float = 0.9,
        system_prompt: str = "You are a skilled coding assistant."
    ) -> Dict:
        """Export synthetic tasks to NeMo-compatible JSONL format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle and split
        import random
        shuffled = tasks.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_tasks = shuffled[:split_idx]
        val_tasks = shuffled[split_idx:]

        # Convert to NeMo format
        def to_nemo(task: SyntheticTask) -> Dict:
            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task.prompt},
                    {"role": "assistant", "content": f"I'll help you {task.prompt.lower()}..."}
                ],
                "metadata": {
                    "synthetic": True,
                    "strategy": "trace_seeded",
                    "task_type": task.task_type,
                    "repo": task.repo,
                    "task_id": task.task_id,
                    "generated_at": datetime.now().isoformat()
                }
            }

        # Write train.jsonl
        with open(output_dir / "train.jsonl", "w") as f:
            for task in train_tasks:
                f.write(json.dumps(to_nemo(task)) + "\n")

        # Write val.jsonl
        with open(output_dir / "val.jsonl", "w") as f:
            for task in val_tasks:
                f.write(json.dumps(to_nemo(task)) + "\n")

        # Write metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_examples": len(tasks),
            "train_examples": len(train_tasks),
            "val_examples": len(val_tasks),
            "train_ratio": train_ratio,
            "strategy": "trace_seeded"
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py::test_export_to_nemo -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add NeMo JSONL export for synthetic tasks"
```

---

## Task 12: Add Synthetic Generation API Endpoints

**Files:**
- Modify: `bashgym/api/factory_routes.py`
- Test: `tests/api/test_factory_routes.py`

**Step 1: Write the failing test**

```python
# tests/api/test_factory_routes.py (add to existing or create)
import pytest
from fastapi.testclient import TestClient
from bashgym.api.routes import app

client = TestClient(app)

def test_post_synthetic_generate():
    """Should accept synthetic generation request."""
    response = client.post("/api/factory/synthetic/generate", json={
        "strategy": "trace_seeded",
        "repo_filter": "single",
        "selected_repos": ["bashgym"],
        "preset": "quick_test",
        "provider": "nim"
    })

    assert response.status_code in [200, 202]  # OK or Accepted
    data = response.json()
    assert "job_id" in data or "status" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_factory_routes.py::test_post_synthetic_generate -v`
Expected: FAIL with "404 Not Found" (endpoint doesn't exist)

**Step 3: Write minimal implementation**

```python
# bashgym/api/factory_routes.py (add to existing routes)
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid

router = APIRouter(prefix="/api/factory", tags=["factory"])

class SyntheticGenerateRequest(BaseModel):
    strategy: str = "trace_seeded"
    repo_filter: str = "all"
    selected_repos: List[str] = []
    preset: str = "balanced"
    target_examples: Optional[int] = None
    multiplier: Optional[int] = None
    provider: str = "nim"
    merge_mode: str = "mixed"

class SyntheticGenerateResponse(BaseModel):
    job_id: str
    status: str

# In-memory job tracking (would use Redis/DB in production)
generation_jobs = {}

@router.post("/synthetic/generate", response_model=SyntheticGenerateResponse)
async def start_synthetic_generation(
    request: SyntheticGenerateRequest,
    background_tasks: BackgroundTasks
):
    """Start a synthetic data generation job."""
    job_id = f"gen_{uuid.uuid4().hex[:8]}"

    generation_jobs[job_id] = {
        "status": "queued",
        "progress": {"current": 0, "total": 0},
        "config": request.model_dump()
    }

    # Add background task
    background_tasks.add_task(run_generation_job, job_id, request)

    return SyntheticGenerateResponse(job_id=job_id, status="queued")

async def run_generation_job(job_id: str, request: SyntheticGenerateRequest):
    """Background task to run generation."""
    from bashgym.factory.synthetic_generator import SyntheticGenerator, PRESETS
    from bashgym.factory.pattern_extractor import PatternExtractor

    try:
        generation_jobs[job_id]["status"] = "running"

        # Load traces based on repo filter
        # TODO: Implement actual trace loading
        traces = []  # Load from data/gold_traces/

        # Extract patterns
        extractor = PatternExtractor()
        repo_name = request.selected_repos[0] if request.selected_repos else "all"
        patterns = extractor.extract_patterns(traces, repo_name=repo_name)

        # Calculate target
        preset = PRESETS.get(request.preset)
        target = request.target_examples or (preset.target_examples if preset else 500)

        generation_jobs[job_id]["progress"]["total"] = target

        # Generate
        generator = SyntheticGenerator()

        def on_progress(current: int, total: int):
            generation_jobs[job_id]["progress"]["current"] = current

        tasks = await generator.generate_batch(
            patterns=patterns,
            seed_prompts=[],  # Extract from traces
            count=target,
            provider=request.provider,
            on_progress=on_progress
        )

        # Export
        from pathlib import Path
        output_dir = Path(f"data/synthetic/{job_id}")
        generator.export_to_nemo(tasks, output_dir)

        generation_jobs[job_id]["status"] = "completed"
        generation_jobs[job_id]["output_dir"] = str(output_dir)

    except Exception as e:
        generation_jobs[job_id]["status"] = "failed"
        generation_jobs[job_id]["error"] = str(e)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_factory_routes.py::test_post_synthetic_generate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/api/factory_routes.py tests/api/test_factory_routes.py
git commit -m "feat(api): add synthetic generation endpoint"
```

---

## Task 13: Add Job Status Endpoint

**Files:**
- Modify: `bashgym/api/factory_routes.py`
- Test: `tests/api/test_factory_routes.py`

**Step 1: Write the failing test**

```python
# tests/api/test_factory_routes.py (add to existing)
def test_get_synthetic_job_status():
    """Should return job status."""
    # First create a job
    create_response = client.post("/api/factory/synthetic/generate", json={
        "strategy": "trace_seeded",
        "preset": "quick_test"
    })
    job_id = create_response.json()["job_id"]

    # Then check status
    response = client.get(f"/api/factory/synthetic/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "progress" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_factory_routes.py::test_get_synthetic_job_status -v`
Expected: FAIL with "404 Not Found"

**Step 3: Write minimal implementation**

```python
# bashgym/api/factory_routes.py (add to existing)
from fastapi import HTTPException

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: dict
    patterns_extracted: int = 0
    examples_generated: int = 0
    error: Optional[str] = None
    output_dir: Optional[str] = None

@router.get("/synthetic/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a synthetic generation job."""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = generation_jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", {"current": 0, "total": 0}),
        patterns_extracted=job.get("patterns_extracted", 0),
        examples_generated=job.get("progress", {}).get("current", 0),
        error=job.get("error"),
        output_dir=job.get("output_dir")
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_factory_routes.py::test_get_synthetic_job_status -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/api/factory_routes.py tests/api/test_factory_routes.py
git commit -m "feat(api): add job status endpoint for synthetic generation"
```

---

## Task 14: Create Frontend SyntheticGenerator Component

**Files:**
- Create: `frontend/src/components/factory/SyntheticGenerator.tsx`
- Modify: `frontend/src/api.ts`

**Step 1: Add API types and functions**

```typescript
// frontend/src/api.ts (add to existing)

// Synthetic generation types
export interface SyntheticGenerateRequest {
  strategy: 'trace_seeded' | 'augmented' | 'schema_driven';
  repo_filter: 'single' | 'multi' | 'all';
  selected_repos: string[];
  preset: 'quick_test' | 'balanced' | 'production' | 'custom';
  target_examples?: number;
  multiplier?: number;
  provider: 'nim' | 'anthropic';
  merge_mode: 'synthetic_only' | 'mixed' | 'synthetic_weighted';
}

export interface SyntheticJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: { current: number; total: number };
  patterns_extracted: number;
  examples_generated: number;
  error?: string;
  output_dir?: string;
}

export const syntheticApi = {
  generate: (request: SyntheticGenerateRequest) =>
    fetch('/api/factory/synthetic/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    }).then(r => r.json()),

  getJobStatus: (jobId: string) =>
    fetch(`/api/factory/synthetic/jobs/${jobId}`).then(r => r.json()),
};
```

**Step 2: Create component**

```typescript
// frontend/src/components/factory/SyntheticGenerator.tsx
import React, { useState } from 'react';
import { syntheticApi, SyntheticGenerateRequest, SyntheticJobStatus } from '../../api';

const PRESETS = {
  quick_test: { label: 'Quick Test', target: 100, description: 'Fast iteration' },
  balanced: { label: 'Balanced', target: 500, description: 'Recommended for LoRA' },
  production: { label: 'Production', target: 2000, description: 'Best results' },
  custom: { label: 'Custom', target: null, description: 'Set your own' },
};

export function SyntheticGenerator() {
  const [strategy, setStrategy] = useState<'trace_seeded' | 'augmented' | 'schema_driven'>('trace_seeded');
  const [repoFilter, setRepoFilter] = useState<'single' | 'multi' | 'all'>('all');
  const [selectedRepos, setSelectedRepos] = useState<string[]>([]);
  const [preset, setPreset] = useState<keyof typeof PRESETS>('balanced');
  const [customTarget, setCustomTarget] = useState(500);
  const [customMultiplier, setCustomMultiplier] = useState(6);
  const [provider, setProvider] = useState<'nim' | 'anthropic'>('nim');
  const [mergeMode, setMergeMode] = useState<'synthetic_only' | 'mixed' | 'synthetic_weighted'>('mixed');

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<SyntheticJobStatus | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const seedCount = 84; // TODO: Fetch from API
  const targetExamples = preset === 'custom' ? customTarget : PRESETS[preset].target!;
  const autoMultiplier = Math.ceil(targetExamples / seedCount);
  const estimatedOutput = seedCount * (preset === 'custom' ? customMultiplier : autoMultiplier);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const request: SyntheticGenerateRequest = {
        strategy,
        repo_filter: repoFilter,
        selected_repos: selectedRepos,
        preset,
        target_examples: preset === 'custom' ? customTarget : undefined,
        multiplier: preset === 'custom' ? customMultiplier : undefined,
        provider,
        merge_mode: mergeMode,
      };

      const response = await syntheticApi.generate(request);
      setJobId(response.job_id);

      // Poll for status
      const pollStatus = async () => {
        const status = await syntheticApi.getJobStatus(response.job_id);
        setJobStatus(status);

        if (status.status === 'running' || status.status === 'queued') {
          setTimeout(pollStatus, 2000);
        } else {
          setIsGenerating(false);
        }
      };
      pollStatus();

    } catch (error) {
      console.error('Generation failed:', error);
      setIsGenerating(false);
    }
  };

  return (
    <div className="p-6 bg-gray-900 rounded-lg">
      <h2 className="text-xl font-bold mb-4">Synthetic Data Generator</h2>

      {/* Strategy Selection */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Strategy</label>
        <div className="flex gap-2">
          {(['trace_seeded', 'augmented', 'schema_driven'] as const).map(s => (
            <button
              key={s}
              onClick={() => setStrategy(s)}
              className={`px-4 py-2 rounded ${strategy === s ? 'bg-blue-600' : 'bg-gray-700'}`}
            >
              {s.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
            </button>
          ))}
        </div>
      </div>

      {/* Repo Filter */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Repo Filter</label>
        <div className="flex gap-4">
          {(['all', 'multi', 'single'] as const).map(rf => (
            <label key={rf} className="flex items-center gap-2">
              <input
                type="radio"
                checked={repoFilter === rf}
                onChange={() => setRepoFilter(rf)}
              />
              {rf.charAt(0).toUpperCase() + rf.slice(1)}
            </label>
          ))}
        </div>
      </div>

      {/* Preset Selection */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Preset</label>
        <div className="flex gap-2">
          {Object.entries(PRESETS).map(([key, val]) => (
            <button
              key={key}
              onClick={() => setPreset(key as keyof typeof PRESETS)}
              className={`px-4 py-2 rounded ${preset === key ? 'bg-blue-600' : 'bg-gray-700'}`}
            >
              {val.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="mb-4 p-4 bg-gray-800 rounded">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Target Examples:</span>{' '}
            {preset === 'custom' ? (
              <input
                type="number"
                value={customTarget}
                onChange={e => setCustomTarget(Number(e.target.value))}
                className="w-20 bg-gray-700 px-2 py-1 rounded"
              />
            ) : (
              targetExamples
            )}
          </div>
          <div>
            <span className="text-gray-400">Seeds Available:</span> {seedCount}
          </div>
          <div>
            <span className="text-gray-400">Multiplier:</span>{' '}
            {preset === 'custom' ? (
              <input
                type="number"
                value={customMultiplier}
                onChange={e => setCustomMultiplier(Number(e.target.value))}
                className="w-20 bg-gray-700 px-2 py-1 rounded"
              />
            ) : (
              `${autoMultiplier}x (auto)`
            )}
          </div>
          <div>
            <span className="text-gray-400">Est. Output:</span> ~{estimatedOutput} examples
          </div>
        </div>
      </div>

      {/* Provider & Merge Mode */}
      <div className="mb-4 grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Provider</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={provider === 'nim'}
                onChange={() => setProvider('nim')}
              />
              NIM (qwen-32b)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={provider === 'anthropic'}
                onChange={() => setProvider('anthropic')}
              />
              Anthropic
            </label>
          </div>
        </div>
        <div>
          <label className="block text-sm text-gray-400 mb-2">Merge Mode</label>
          <div className="flex gap-4">
            {(['synthetic_only', 'mixed', 'synthetic_weighted'] as const).map(mm => (
              <label key={mm} className="flex items-center gap-2">
                <input
                  type="radio"
                  checked={mergeMode === mm}
                  onChange={() => setMergeMode(mm)}
                />
                {mm.replace('_', ' ')}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Progress */}
      {jobStatus && (
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <div className="flex justify-between mb-2">
            <span>Status: {jobStatus.status}</span>
            <span>{jobStatus.progress.current}/{jobStatus.progress.total}</span>
          </div>
          <div className="w-full bg-gray-700 rounded h-2">
            <div
              className="bg-blue-600 h-2 rounded"
              style={{
                width: `${(jobStatus.progress.current / jobStatus.progress.total) * 100}%`
              }}
            />
          </div>
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded font-bold"
      >
        {isGenerating ? 'Generating...' : 'Generate Synthetic Dataset'}
      </button>
    </div>
  );
}
```

**Step 3: Run frontend to verify it compiles**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/factory/SyntheticGenerator.tsx frontend/src/api.ts
git commit -m "feat(frontend): add SyntheticGenerator component with full UI"
```

---

## Task 15: Wire Augmented Strategy (Existing Infrastructure)

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
@pytest.mark.asyncio
async def test_augmented_strategy():
    """Should generate variations using augmented strategy."""
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_augmented_strategy -v`
Expected: FAIL with "SyntheticGenerator has no attribute 'generate_augmented'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add to SyntheticGenerator class)
    async def generate_augmented(
        self,
        seed_examples: List[Dict],
        variations_per_seed: int = 3,
        provider: str = "nim"
    ) -> List[SyntheticTask]:
        """Generate variations of existing prompts (augmented strategy)."""
        tasks = []

        for seed in seed_examples:
            prompt = seed.get("prompt", "")

            for i in range(variations_per_seed):
                augment_prompt = f"""Rewrite this coding task to create a variation:
Original: "{prompt}"

Create a similar but different task that:
- Has the same complexity level
- Targets similar types of files
- Uses different specific details

Return ONLY the rewritten task, nothing else."""

                try:
                    variation = await self._call_llm(augment_prompt, provider)
                    tasks.append(SyntheticTask(
                        task_id=f"aug_{uuid.uuid4().hex[:8]}",
                        prompt=variation,
                        target_files=[],
                        task_type="feature",
                        expected_tools=["Read", "Edit"],
                        source_pattern_id=seed.get("id", ""),
                        repo=seed.get("repo", ""),
                        metadata={"strategy": "augmented", "seed_prompt": prompt}
                    ))
                except Exception as e:
                    print(f"Augmentation failed: {e}")

        return tasks
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py::test_augmented_strategy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add augmented generation strategy"
```

---

## Task 16: Wire Schema-Driven Strategy (Existing Infrastructure)

**Files:**
- Modify: `bashgym/factory/synthetic_generator.py`
- Test: `tests/factory/test_synthetic_generator.py`

**Step 1: Write the failing test**

```python
# tests/factory/test_synthetic_generator.py (add to existing)
@pytest.mark.asyncio
async def test_schema_driven_strategy():
    """Should generate tasks from repo schema."""
    generator = SyntheticGenerator()

    repo_schema = {
        "name": "bashgym",
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
        assert all(t.repo == "bashgym" for t in tasks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factory/test_synthetic_generator.py::test_schema_driven_strategy -v`
Expected: FAIL with "SyntheticGenerator has no attribute 'generate_from_schema'"

**Step 3: Write minimal implementation**

```python
# bashgym/factory/synthetic_generator.py (add to SyntheticGenerator class)
    async def generate_from_schema(
        self,
        repo_schema: Dict,
        count: int = 10,
        provider: str = "nim"
    ) -> List[SyntheticTask]:
        """Generate tasks from repo structure schema (schema-driven strategy)."""
        tasks = []

        repo_name = repo_schema.get("name", "unknown")
        structure = repo_schema.get("structure", {})
        frameworks = repo_schema.get("frameworks", [])

        # Build structure description
        structure_desc = "\n".join(
            f"- {dir}: {', '.join(files)}"
            for dir, files in structure.items()
        )

        for i in range(count):
            schema_prompt = f"""Generate a coding task for a project with this structure:

Project: {repo_name}
Frameworks: {', '.join(frameworks)}
Structure:
{structure_desc}

Generate a realistic development task that would involve 1-3 of these files.
Return ONLY the task prompt, nothing else."""

            try:
                task_prompt = await self._call_llm(schema_prompt, provider)
                tasks.append(SyntheticTask(
                    task_id=f"schema_{uuid.uuid4().hex[:8]}",
                    prompt=task_prompt,
                    target_files=[],
                    task_type="feature",
                    expected_tools=["Read", "Edit", "Bash"],
                    source_pattern_id="schema",
                    repo=repo_name,
                    metadata={"strategy": "schema_driven"}
                ))
            except Exception as e:
                print(f"Schema generation failed: {e}")

        return tasks
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factory/test_synthetic_generator.py::test_schema_driven_strategy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/factory/synthetic_generator.py tests/factory/test_synthetic_generator.py
git commit -m "feat(factory): add schema-driven generation strategy"
```

---

## Task 17: Integration Test - Full Pipeline

**Files:**
- Create: `tests/factory/test_synthetic_integration.py`

**Step 1: Write the integration test**

```python
# tests/factory/test_synthetic_integration.py
import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
import tempfile
import json

from bashgym.factory.pattern_extractor import PatternExtractor
from bashgym.factory.synthetic_generator import SyntheticGenerator, PRESETS

@pytest.mark.asyncio
async def test_full_synthetic_pipeline():
    """Integration test: extract patterns → generate tasks → export."""

    # 1. Create mock traces
    traces = [
        {
            "metadata": {"repo": "bashgym"},
            "summary": {"task_description": "Add retry logic to API"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/api.py"}},
                {"tool": "Edit", "input": {"file_path": "src/api.py"}},
                {"tool": "Bash", "input": {"command": "pytest"}},
            ]
        },
        {
            "metadata": {"repo": "bashgym"},
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
    patterns = extractor.extract_patterns(traces, repo_name="bashgym")

    assert patterns.repo_name == "bashgym"
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
        assert all(t.repo == "bashgym" for t in tasks)

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
```

**Step 2: Run integration test**

Run: `pytest tests/factory/test_synthetic_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/factory/test_synthetic_integration.py
git commit -m "test(factory): add full synthetic pipeline integration test"
```

---

## Summary

This implementation plan creates the Synthetic Data Generator with:

| Task | Description | Files |
|------|-------------|-------|
| 1-5 | PatternExtractor (data structures, classification, extraction) | `pattern_extractor.py` |
| 6-11 | SyntheticGenerator (generation, NIM integration, export) | `synthetic_generator.py` |
| 12-13 | API endpoints (generate, status) | `factory_routes.py` |
| 14 | Frontend UI component | `SyntheticGenerator.tsx` |
| 15-16 | Augmented and Schema-driven strategies | `synthetic_generator.py` |
| 17 | Integration test | `test_synthetic_integration.py` |

**Total estimated tasks:** 17 bite-sized tasks (2-5 min each)

**Dependencies:** None external (uses existing NIM/Anthropic client patterns)

---

**Plan complete and saved to `docs/plans/2026-01-25-synthetic-data-generator-implementation.md`.**

Two execution options:

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
