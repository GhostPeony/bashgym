"""
Embedding-Based Deduplication for Training Data

Uses NVIDIA NIM API embeddings for semantic deduplication of training examples.
Removes near-duplicate examples that would waste training compute without
adding diversity. Computes diversity scores for data quality monitoring.

Module 3: Data Synthesis (The "Factory") - Dedup
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# NIM embedding model (cost-effective, good quality)
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1"


@dataclass
class DedupConfig:
    """Configuration for embedding-based deduplication."""

    similarity_threshold: float = 0.95  # Cosine similarity above this = duplicate
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    nim_endpoint: str = DEFAULT_NIM_ENDPOINT
    nim_api_key: str | None = None
    batch_size: int = 32  # Embeddings per API call
    timeout: float = 30.0  # API timeout in seconds
    max_retries: int = 2

    def __post_init__(self):
        if not self.nim_api_key:
            self.nim_api_key = os.environ.get("NVIDIA_API_KEY")


@dataclass
class DedupResult:
    """Result of deduplication."""

    original_count: int
    deduplicated_count: int
    duplicates_removed: int
    diversity_score: float  # 0.0 = all identical, 1.0 = all unique
    duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)


class EmbeddingDeduplicator:
    """Semantic deduplication using NIM API embeddings.

    Computes embeddings for training examples, finds near-duplicates
    by cosine similarity, and removes them. Gracefully falls back
    to no-op when NIM API is unavailable.
    """

    def __init__(self, config: DedupConfig | None = None):
        self.config = config or DedupConfig()
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.nim_endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.nim_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    def compute_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts via NIM API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            RuntimeError: If NIM API is unavailable after retries
        """
        if not self.config.nim_api_key:
            raise RuntimeError("NIM API key not configured for embeddings")

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            for attempt in range(self.config.max_retries + 1):
                try:
                    response = self.client.post(
                        "/embeddings",
                        json={
                            "model": self.config.embedding_model,
                            "input": batch,
                            "input_type": "passage",
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Sort by index to maintain order
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    batch_embeddings = [item["embedding"] for item in sorted_data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except (httpx.HTTPError, KeyError) as e:
                    if attempt == self.config.max_retries:
                        raise RuntimeError(
                            f"NIM embedding API failed after {self.config.max_retries + 1} attempts: {e}"
                        )
                    logger.warning(
                        "NIM embedding attempt %d failed: %s, retrying...",
                        attempt + 1,
                        e,
                    )

        return all_embeddings

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def find_duplicates(
        self,
        embeddings: list[list[float]],
        threshold: float | None = None,
    ) -> list[tuple[int, int]]:
        """Find pairs of near-duplicate examples by cosine similarity.

        Args:
            embeddings: List of embedding vectors
            threshold: Similarity threshold (default: config.similarity_threshold)

        Returns:
            List of (index_a, index_b) pairs where similarity > threshold
        """
        thresh = threshold or self.config.similarity_threshold
        duplicates: list[tuple[int, int]] = []
        n = len(embeddings)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= thresh:
                    duplicates.append((i, j))

        return duplicates

    def deduplicate(self, examples: list[dict[str, Any]]) -> DedupResult:
        """Deduplicate training examples using semantic embeddings.

        Extracts text from examples, computes embeddings, finds duplicates,
        and returns a DedupResult with the filtered list.

        Args:
            examples: List of training example dicts (must have 'messages' or 'prompt' key)

        Returns:
            DedupResult with deduplicated examples and metrics
        """
        if not examples:
            return DedupResult(
                original_count=0,
                deduplicated_count=0,
                duplicates_removed=0,
                diversity_score=1.0,
            )

        # Extract text for embedding
        texts = [self._extract_text(ex) for ex in examples]

        try:
            embeddings = self.compute_embeddings(texts)
        except RuntimeError as e:
            logger.warning("Embedding dedup skipped: %s", e)
            return DedupResult(
                original_count=len(examples),
                deduplicated_count=len(examples),
                duplicates_removed=0,
                diversity_score=-1.0,  # -1 indicates dedup was skipped
            )

        # Find duplicates
        duplicate_pairs = self.find_duplicates(embeddings)

        # Build set of indices to remove (keep the first in each pair)
        to_remove: set[int] = set()
        for _, j in duplicate_pairs:
            to_remove.add(j)

        # Filter
        deduplicated = [ex for i, ex in enumerate(examples) if i not in to_remove]

        # Compute diversity score
        div_score = self.diversity_score(embeddings)

        return DedupResult(
            original_count=len(examples),
            deduplicated_count=len(deduplicated),
            duplicates_removed=len(to_remove),
            diversity_score=div_score,
            duplicate_pairs=duplicate_pairs,
        )

    def diversity_score(self, embeddings: list[list[float]]) -> float:
        """Compute a diversity score for a set of embeddings.

        Score is 1 - (average pairwise similarity). Higher = more diverse.
        Range: 0.0 (all identical) to 1.0 (all orthogonal).

        For efficiency, samples up to 100 random pairs for large sets.
        """
        import random

        n = len(embeddings)
        if n < 2:
            return 1.0

        # Sample pairs for efficiency
        max_pairs = 100
        if n * (n - 1) // 2 <= max_pairs:
            # Compute all pairs
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs = []
            seen: set[tuple[int, int]] = set()
            while len(pairs) < max_pairs:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j and (i, j) not in seen and (j, i) not in seen:
                    pairs.append((i, j))
                    seen.add((i, j))

        total_sim = sum(self._cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs)
        avg_sim = total_sim / len(pairs)

        return max(0.0, min(1.0, 1.0 - avg_sim))

    def _extract_text(self, example: dict[str, Any]) -> str:
        """Extract representative text from a training example for embedding."""
        # Try messages format first
        messages = example.get("messages", [])
        if messages:
            parts = []
            for msg in messages:
                content = msg.get("content", "")
                if content and msg.get("role") in ("user", "assistant"):
                    parts.append(content[:500])  # Cap per message
            return " ".join(parts)[:2000]  # Cap total

        # Try prompt/response format
        prompt = example.get("prompt", example.get("instruction", ""))
        response = example.get("response", example.get("chosen", ""))
        text = f"{prompt} {response}".strip()
        return text[:2000]

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
