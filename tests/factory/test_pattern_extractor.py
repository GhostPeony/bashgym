# tests/factory/test_pattern_extractor.py
"""Tests for pattern extraction data structures."""

import pytest
from bashgym.factory.pattern_extractor import TracePatterns, FileCluster, ToolSequence, PatternExtractor


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
        repo_name="ghostwork",
        framework_hints=["fastapi", "pytest"]
    )

    assert patterns.task_types["feature"] == 0.4
    assert len(patterns.file_clusters) == 1
    assert patterns.repo_name == "ghostwork"


def test_file_cluster_dataclass():
    """FileCluster should store glob patterns and frequency."""
    cluster = FileCluster(patterns=["src/*.py", "tests/*.py"], frequency=0.6)

    assert cluster.patterns == ["src/*.py", "tests/*.py"]
    assert cluster.frequency == 0.6


def test_tool_sequence_dataclass():
    """ToolSequence should store tool names and frequency."""
    sequence = ToolSequence(tools=["Read", "Edit", "Bash"], frequency=0.5)

    assert sequence.tools == ["Read", "Edit", "Bash"]
    assert sequence.frequency == 0.5


def test_trace_patterns_defaults():
    """TracePatterns should have sensible defaults for all fields."""
    patterns = TracePatterns()

    assert patterns.task_types == {}
    assert patterns.file_clusters == []
    assert patterns.common_paths == []
    assert patterns.languages == {}
    assert patterns.tool_patterns == []
    assert patterns.avg_tool_calls == 0
    assert patterns.prompt_templates == []
    assert patterns.prompt_keywords == []
    assert patterns.repo_name == ""
    assert patterns.framework_hints == []


# =============================================================================
# PatternExtractor._classify_task_type() tests
# =============================================================================


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


def test_classify_task_type_default():
    """Should default to 'feature' when no keywords match."""
    extractor = PatternExtractor()
    assert extractor._classify_task_type("Hello world") == "feature"
    assert extractor._classify_task_type("") == "feature"


def test_classify_task_type_case_insensitive():
    """Should be case insensitive."""
    extractor = PatternExtractor()
    assert extractor._classify_task_type("FIX THE BUG") == "bugfix"
    assert extractor._classify_task_type("ADD a Feature") == "feature"
    assert extractor._classify_task_type("REFACTOR the code") == "refactor"


def test_classify_task_type_highest_score_wins():
    """When multiple task types match, the one with highest score wins."""
    extractor = PatternExtractor()
    # "fix bug error" has 3 bugfix keywords vs "add" which has 1 feature keyword
    assert extractor._classify_task_type("Fix the bug error, then add logging") == "bugfix"


# =============================================================================
# PatternExtractor._detect_languages() tests
# =============================================================================


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


def test_detect_languages_empty_list():
    """Should return empty dict for empty file list."""
    extractor = PatternExtractor()
    languages = extractor._detect_languages([])
    assert languages == {}


def test_detect_languages_unknown_extensions():
    """Should classify unknown extensions as 'other'."""
    extractor = PatternExtractor()
    files = ["file.xyz", "data.csv", "image.png"]
    languages = extractor._detect_languages(files)
    assert languages["other"] == 1.0


def test_detect_languages_mixed_case_extensions():
    """Should handle mixed case extensions."""
    extractor = PatternExtractor()
    files = ["Main.PY", "utils.Py", "Test.py"]
    languages = extractor._detect_languages(files)
    assert languages["python"] == 1.0


def test_detect_languages_all_supported():
    """Should detect all supported languages."""
    extractor = PatternExtractor()
    files = [
        "main.py",      # python
        "app.ts",       # typescript
        "comp.tsx",     # typescript
        "script.js",    # javascript
        "comp.jsx",     # javascript
        "README.md",    # markdown
        "config.json",  # json
        "config.yaml",  # yaml
        "config.yml",   # yaml
        "setup.sh",     # bash
        "main.rs",      # rust
        "main.go",      # go
    ]
    languages = extractor._detect_languages(files)

    # Verify all languages are detected
    assert "python" in languages
    assert "typescript" in languages
    assert "javascript" in languages
    assert "markdown" in languages
    assert "json" in languages
    assert "yaml" in languages
    assert "bash" in languages
    assert "rust" in languages
    assert "go" in languages


# =============================================================================
# PatternExtractor._extract_tool_sequences() tests
# =============================================================================


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


def test_extract_tool_sequences_empty_list():
    """Should return empty list for empty input."""
    extractor = PatternExtractor()
    patterns = extractor._extract_tool_sequences([], n=3, min_frequency=0.2)
    assert patterns == []


def test_extract_tool_sequences_below_threshold():
    """Should filter out sequences below min_frequency."""
    extractor = PatternExtractor()

    tool_sequences = [
        ["Read", "Edit", "Bash"],
        ["Glob", "Read", "Write"],
        ["Read", "Grep", "Edit"],
        ["Bash", "Read", "Edit"],
        ["Write", "Bash", "Read"],
    ]

    # With min_frequency=0.5, only sequences appearing 3+ times should be included
    patterns = extractor._extract_tool_sequences(tool_sequences, n=3, min_frequency=0.5)
    assert patterns == []  # No sequence appears 3 or more times


def test_extract_tool_sequences_different_n():
    """Should work with different N-gram sizes."""
    extractor = PatternExtractor()

    tool_sequences = [
        ["Read", "Edit", "Bash", "Read"],
        ["Read", "Edit", "Bash", "Write"],
        ["Read", "Edit", "Bash", "Read"],
    ]

    # Extract 2-grams
    patterns_2 = extractor._extract_tool_sequences(tool_sequences, n=2, min_frequency=0.3)
    assert any(p.tools == ["Read", "Edit"] for p in patterns_2)
    assert any(p.tools == ["Edit", "Bash"] for p in patterns_2)

    # Extract 4-grams
    patterns_4 = extractor._extract_tool_sequences(tool_sequences, n=4, min_frequency=0.3)
    assert any(p.tools == ["Read", "Edit", "Bash", "Read"] for p in patterns_4)


def test_extract_tool_sequences_sorted_by_frequency():
    """Should return patterns sorted by frequency (most common first)."""
    extractor = PatternExtractor()

    tool_sequences = [
        ["Read", "Edit", "Bash"],
        ["Read", "Edit", "Bash"],
        ["Read", "Edit", "Bash"],
        ["Glob", "Read", "Edit"],
        ["Glob", "Read", "Edit"],
    ]

    patterns = extractor._extract_tool_sequences(tool_sequences, n=3, min_frequency=0.2)

    # "Read, Edit, Bash" (0.6) should come before "Glob, Read, Edit" (0.4)
    assert len(patterns) >= 2
    assert patterns[0].tools == ["Read", "Edit", "Bash"]
    assert patterns[0].frequency == pytest.approx(0.6, rel=0.01)
    assert patterns[1].tools == ["Glob", "Read", "Edit"]
    assert patterns[1].frequency == pytest.approx(0.4, rel=0.01)


def test_extract_tool_sequences_short_sequences():
    """Should handle sequences shorter than n."""
    extractor = PatternExtractor()

    tool_sequences = [
        ["Read", "Edit"],  # Too short for n=3
        ["Read", "Edit", "Bash"],
        ["Read", "Edit", "Bash"],
    ]

    patterns = extractor._extract_tool_sequences(tool_sequences, n=3, min_frequency=0.3)

    # Should still work, ignoring the short sequence
    assert any(p.tools == ["Read", "Edit", "Bash"] for p in patterns)


# =============================================================================
# PatternExtractor.extract_patterns() tests
# =============================================================================


def test_extract_patterns_from_traces():
    """Should extract patterns from a list of trace dicts."""
    extractor = PatternExtractor()

    # Mock trace data (simplified structure matching actual traces)
    traces = [
        {
            "metadata": {"repo": "ghostwork"},
            "summary": {"task_description": "Add retry logic to API client"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/api.py"}},
                {"tool": "Edit", "input": {"file_path": "src/api.py"}},
                {"tool": "Bash", "input": {"command": "pytest tests/"}},
            ]
        },
        {
            "metadata": {"repo": "ghostwork"},
            "summary": {"task_description": "Fix bug in authentication"},
            "trace": [
                {"tool": "Glob", "input": {"pattern": "src/*.py"}},
                {"tool": "Read", "input": {"file_path": "src/auth.py"}},
                {"tool": "Edit", "input": {"file_path": "src/auth.py"}},
            ]
        },
    ]

    patterns = extractor.extract_patterns(traces, repo_name="ghostwork")

    assert patterns.repo_name == "ghostwork"
    assert "feature" in patterns.task_types or "bugfix" in patterns.task_types
    assert patterns.avg_tool_calls == 3
    assert "python" in patterns.languages


def test_extract_patterns_empty_traces():
    """Should return empty patterns for empty trace list."""
    extractor = PatternExtractor()
    patterns = extractor.extract_patterns([], repo_name="test-repo")

    assert patterns.repo_name == "test-repo"
    assert patterns.task_types == {}
    assert patterns.languages == {}
    assert patterns.avg_tool_calls == 0


def test_extract_patterns_extracts_common_paths():
    """Should extract common directory paths from file accesses."""
    extractor = PatternExtractor()

    traces = [
        {
            "summary": {"task_description": "Update files"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/api/client.py"}},
                {"tool": "Read", "input": {"file_path": "src/api/server.py"}},
                {"tool": "Edit", "input": {"file_path": "src/api/utils.py"}},
            ]
        },
        {
            "summary": {"task_description": "More updates"},
            "trace": [
                {"tool": "Read", "input": {"file_path": "src/api/models.py"}},
                {"tool": "Edit", "input": {"file_path": "src/api/routes.py"}},
            ]
        },
    ]

    patterns = extractor.extract_patterns(traces)

    # "src/api/" should appear as a common path
    assert any("src/api" in p for p in patterns.common_paths)


def test_extract_patterns_extracts_keywords():
    """Should extract action keywords from prompts."""
    extractor = PatternExtractor()

    traces = [
        {"summary": {"task_description": "Add new feature"}, "trace": []},
        {"summary": {"task_description": "Fix the bug"}, "trace": []},
        {"summary": {"task_description": "Implement logging"}, "trace": []},
    ]

    patterns = extractor.extract_patterns(traces)

    # Should have extracted keywords from the prompts
    assert "add" in patterns.prompt_keywords or "fix" in patterns.prompt_keywords


def test_extract_patterns_handles_path_in_input():
    """Should extract file paths from both 'file_path' and 'path' input fields."""
    extractor = PatternExtractor()

    traces = [
        {
            "summary": {"task_description": "Search files"},
            "trace": [
                {"tool": "Glob", "input": {"path": "src/", "pattern": "*.py"}},
                {"tool": "Grep", "input": {"path": "src/main.py", "pattern": "def"}},
            ]
        },
    ]

    patterns = extractor.extract_patterns(traces)

    assert "python" in patterns.languages


# =============================================================================
# PatternExtractor._extract_common_paths() tests
# =============================================================================


def test_extract_common_paths():
    """Should extract common directory prefixes."""
    extractor = PatternExtractor()

    file_paths = [
        "src/api/client.py",
        "src/api/server.py",
        "src/api/models.py",
        "src/utils.py",
        "tests/test_api.py",
    ]

    common_paths = extractor._extract_common_paths(file_paths)

    # "src/" should be common (appears in 4/5 files)
    assert any("src/" in p for p in common_paths)


def test_extract_common_paths_empty():
    """Should return empty list for no file paths."""
    extractor = PatternExtractor()
    assert extractor._extract_common_paths([]) == []


def test_extract_common_paths_threshold():
    """Should only include paths that meet frequency threshold."""
    extractor = PatternExtractor()

    # Only one file in "rare/" - should not meet 20% threshold
    file_paths = [
        "src/a.py",
        "src/b.py",
        "src/c.py",
        "src/d.py",
        "rare/one.py",
    ]

    common_paths = extractor._extract_common_paths(file_paths)

    # "src/" should be included, "rare/" should not
    assert any("src/" in p for p in common_paths)
    assert not any("rare/" in p for p in common_paths)


# =============================================================================
# PatternExtractor._extract_keywords() tests
# =============================================================================


def test_extract_keywords():
    """Should extract action keywords from prompts."""
    extractor = PatternExtractor()

    prompts = [
        "Add a new login feature",
        "Fix the authentication bug",
        "Refactor the utils module",
    ]

    keywords = extractor._extract_keywords(prompts)

    assert "add" in keywords
    assert "fix" in keywords
    assert "refactor" in keywords


def test_extract_keywords_empty():
    """Should return empty list for no prompts."""
    extractor = PatternExtractor()
    assert extractor._extract_keywords([]) == []


def test_extract_keywords_no_matches():
    """Should return empty list when no keywords match."""
    extractor = PatternExtractor()
    keywords = extractor._extract_keywords(["hello world", "foo bar"])
    assert keywords == []
