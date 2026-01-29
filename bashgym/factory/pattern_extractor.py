# bashgym/factory/pattern_extractor.py
"""Pattern extraction from execution traces for synthetic data generation."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, ClassVar


@dataclass
class FileCluster:
    """Files that are frequently modified together."""
    patterns: List[str] = field(default_factory=list)  # Glob patterns like ["src/*.py", "tests/*.py"]
    frequency: float = 0.0  # How often this cluster appears (0-1)


@dataclass
class ToolSequence:
    """Common tool call sequences."""
    tools: List[str] = field(default_factory=list)  # ["Read", "Edit", "Bash"]
    frequency: float = 0.0  # How often this sequence appears (0-1)


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


class PatternExtractor:
    """Extracts patterns from execution traces for synthetic data generation.

    This class analyzes execution traces to identify:
    - Task types (feature, bugfix, refactor, spec)
    - Programming languages used
    - Common tool call sequences
    - File access patterns
    """

    # File extension to language mapping
    EXTENSION_LANGUAGES: ClassVar[Dict[str, str]] = {
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

    # Keywords for task classification
    TASK_KEYWORDS: ClassVar[Dict[str, List[str]]] = {
        "feature": ["add", "implement", "create", "build", "new", "introduce"],
        "bugfix": ["fix", "bug", "error", "debug", "resolve", "crash", "issue", "broken"],
        "refactor": ["refactor", "clean", "improve", "reorganize", "restructure", "simplify"],
        "spec": ["spec", "design", "plan", "architect", "document", "define", "outline"],
    }

    # Priority order for tie-breaking (higher = more specific)
    TASK_PRIORITY: ClassVar[Dict[str, int]] = {
        "feature": 0,  # Most general, lowest priority
        "bugfix": 1,
        "refactor": 2,
        "spec": 3,  # Most specific, highest priority
    }

    def _classify_task_type(self, prompt: str) -> str:
        """Classify a prompt into task type based on keywords.

        Args:
            prompt: The task description or user prompt to classify.

        Returns:
            One of: "feature", "bugfix", "refactor", "spec".
            Defaults to "feature" if no keywords match.
        """
        prompt_lower = prompt.lower()

        # Count keyword matches for each type
        scores: Dict[str, int] = {}
        for task_type, keywords in self.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return "feature"  # Default to feature

        # Return type with highest score, using priority for tie-breaking
        # Sort by (score, priority) descending
        return max(scores.keys(), key=lambda t: (scores[t], self.TASK_PRIORITY[t]))

    def _detect_languages(self, file_paths: List[str]) -> Dict[str, float]:
        """Detect language distribution from file paths.

        Args:
            file_paths: List of file paths to analyze.

        Returns:
            Dictionary mapping language names to their frequency (0-1).
            Empty dict if no file paths provided.
        """
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

    def _extract_tool_sequences(
        self,
        tool_lists: List[List[str]],
        n: int = 3,
        min_frequency: float = 0.2
    ) -> List[ToolSequence]:
        """Extract common N-gram tool sequences.

        Args:
            tool_lists: List of tool call sequences from traces.
            n: Size of N-grams to extract.
            min_frequency: Minimum frequency threshold (0-1) for inclusion.

        Returns:
            List of ToolSequence objects sorted by frequency (highest first).
        """
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

    def _extract_common_paths(self, file_paths: List[str]) -> List[str]:
        """Extract common directory prefixes from file paths.

        Args:
            file_paths: List of file paths to analyze.

        Returns:
            List of common directory prefixes that appear in at least 20% of files.
        """
        from pathlib import Path

        if not file_paths:
            return []

        dir_counts: Counter = Counter()

        for path in file_paths:
            parts = Path(path).parts
            for i in range(1, len(parts)):
                prefix = "/".join(parts[:i]) + "/"
                dir_counts[prefix] += 1

        threshold = len(file_paths) * 0.2
        return [d for d, c in dir_counts.most_common(10) if c > threshold]

    def _extract_keywords(self, prompts: List[str]) -> List[str]:
        """Extract common action keywords from prompts.

        Args:
            prompts: List of prompt/task description strings.

        Returns:
            List of keywords found in the prompts.
        """
        if not prompts:
            return []

        # Collect all known keywords
        all_keywords = set()
        for keywords in self.TASK_KEYWORDS.values():
            all_keywords.update(keywords)

        # Find keywords that appear in any prompt
        found = set()
        for prompt in prompts:
            prompt_lower = prompt.lower()
            for kw in all_keywords:
                if kw in prompt_lower:
                    found.add(kw)

        return list(found)

    def extract_patterns(self, traces: List[Dict], repo_name: str = "") -> TracePatterns:
        """Extract patterns from a list of trace dictionaries.

        This is the main orchestration method that combines all extraction methods
        to create a comprehensive TracePatterns object.

        Args:
            traces: List of trace dictionaries containing metadata, summary, and trace steps.
            repo_name: Name of the repository for context.

        Returns:
            TracePatterns object with extracted patterns.
        """
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
            file_clusters=[],
            common_paths=common_paths,
            languages=languages,
            tool_patterns=tool_patterns,
            avg_tool_calls=avg_tool_calls,
            prompt_templates=[],
            prompt_keywords=prompt_keywords,
            repo_name=repo_name,
            framework_hints=[]
        )
