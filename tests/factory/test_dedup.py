"""Tests for embedding-based deduplication."""

from unittest.mock import MagicMock, patch

import pytest

from bashgym.factory.dedup import DedupConfig, DedupResult, EmbeddingDeduplicator

# =========================================================================
# DedupConfig
# =========================================================================


class TestDedupConfig:
    def test_defaults(self):
        config = DedupConfig()
        assert config.similarity_threshold == 0.95
        assert config.batch_size == 32
        assert config.timeout == 30.0
        assert config.max_retries == 2
        assert config.embedding_model == "nvidia/nv-embedqa-e5-v5"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = DedupConfig()
            assert config.nim_api_key == "test-key"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "env-key"}):
            config = DedupConfig(nim_api_key="explicit-key")
            assert config.nim_api_key == "explicit-key"

    def test_api_key_none_when_env_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            config = DedupConfig()
            assert config.nim_api_key is None

    def test_custom_settings(self):
        config = DedupConfig(
            similarity_threshold=0.8,
            batch_size=64,
            timeout=60.0,
            max_retries=5,
        )
        assert config.similarity_threshold == 0.8
        assert config.batch_size == 64
        assert config.timeout == 60.0
        assert config.max_retries == 5


# =========================================================================
# DedupResult
# =========================================================================


class TestDedupResult:
    def test_result_fields(self):
        result = DedupResult(
            original_count=100,
            deduplicated_count=85,
            duplicates_removed=15,
            diversity_score=0.72,
        )
        assert result.original_count == 100
        assert result.deduplicated_count == 85
        assert result.duplicates_removed == 15
        assert result.diversity_score == 0.72
        assert result.duplicate_pairs == []

    def test_result_with_pairs(self):
        result = DedupResult(
            original_count=5,
            deduplicated_count=3,
            duplicates_removed=2,
            diversity_score=0.6,
            duplicate_pairs=[(0, 1), (2, 4)],
        )
        assert len(result.duplicate_pairs) == 2
        assert result.duplicate_pairs[0] == (0, 1)


# =========================================================================
# _cosine_similarity
# =========================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        sim = dedup._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        sim = dedup._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert sim == pytest.approx(0.0)

    def test_opposite_vectors(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        sim = dedup._cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert sim == pytest.approx(-1.0)

    def test_zero_vector(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        sim = dedup._cosine_similarity([0, 0, 0], [1, 0, 0])
        assert sim == 0.0

    def test_both_zero_vectors(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        sim = dedup._cosine_similarity([0, 0, 0], [0, 0, 0])
        assert sim == 0.0

    def test_non_unit_vectors(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        # [2, 0, 0] and [3, 0, 0] are parallel -> cosine sim = 1.0
        sim = dedup._cosine_similarity([2, 0, 0], [3, 0, 0])
        assert sim == pytest.approx(1.0)

    def test_similar_vectors(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        sim = dedup._cosine_similarity([1, 0.1, 0], [1, 0, 0])
        assert 0.9 < sim < 1.0


# =========================================================================
# find_duplicates
# =========================================================================


class TestFindDuplicates:
    def test_finds_duplicates(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0]]
        dupes = dedup.find_duplicates(embeddings, threshold=0.95)
        assert len(dupes) == 1
        assert dupes[0] == (0, 1)

    def test_no_duplicates(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        dupes = dedup.find_duplicates(embeddings, threshold=0.95)
        assert len(dupes) == 0

    def test_all_duplicates(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        dupes = dedup.find_duplicates(embeddings, threshold=0.95)
        # (0,1), (0,2), (1,2)
        assert len(dupes) == 3

    def test_empty_embeddings(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        dupes = dedup.find_duplicates([], threshold=0.95)
        assert len(dupes) == 0

    def test_single_embedding(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        dupes = dedup.find_duplicates([[1, 0, 0]], threshold=0.95)
        assert len(dupes) == 0

    def test_uses_config_threshold_as_default(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake", similarity_threshold=0.5))
        # These vectors are similar enough at 0.5 threshold
        embeddings = [[1, 0.5, 0], [1, 0, 0]]
        dupes = dedup.find_duplicates(embeddings)
        assert len(dupes) == 1


# =========================================================================
# deduplicate
# =========================================================================


class TestDeduplicate:
    def test_empty_examples(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        result = dedup.deduplicate([])
        assert result.original_count == 0
        assert result.deduplicated_count == 0
        assert result.duplicates_removed == 0
        assert result.diversity_score == 1.0

    def test_graceful_fallback_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key=None))
        examples = [{"messages": [{"role": "user", "content": "test"}]}]
        result = dedup.deduplicate(examples)
        assert result.diversity_score == -1.0  # Skipped
        assert result.duplicates_removed == 0
        assert result.deduplicated_count == 1

    @patch.object(EmbeddingDeduplicator, "compute_embeddings")
    def test_removes_duplicates(self, mock_embed):
        mock_embed.return_value = [[1, 0, 0], [0.999, 0.001, 0], [0, 1, 0]]
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        examples = [
            {"messages": [{"role": "user", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "hello!"}]},
            {"messages": [{"role": "user", "content": "goodbye"}]},
        ]
        result = dedup.deduplicate(examples)
        assert result.original_count == 3
        assert result.duplicates_removed == 1
        assert result.deduplicated_count == 2

    @patch.object(EmbeddingDeduplicator, "compute_embeddings")
    def test_keeps_first_of_duplicate_pair(self, mock_embed):
        """Should keep the first example and remove the later duplicate."""
        mock_embed.return_value = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        examples = [
            {"messages": [{"role": "user", "content": "first"}]},
            {"messages": [{"role": "user", "content": "duplicate"}]},
            {"messages": [{"role": "user", "content": "different"}]},
        ]
        result = dedup.deduplicate(examples)
        assert result.deduplicated_count == 2
        assert result.duplicate_pairs == [(0, 1)]

    @patch.object(EmbeddingDeduplicator, "compute_embeddings")
    def test_no_duplicates_found(self, mock_embed):
        mock_embed.return_value = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        examples = [
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]},
            {"messages": [{"role": "user", "content": "c"}]},
        ]
        result = dedup.deduplicate(examples)
        assert result.original_count == 3
        assert result.deduplicated_count == 3
        assert result.duplicates_removed == 0

    @patch.object(EmbeddingDeduplicator, "compute_embeddings")
    def test_embedding_api_failure_returns_skipped(self, mock_embed):
        """When embedding API fails, should return skipped result."""
        mock_embed.side_effect = RuntimeError("NIM API unavailable")
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        examples = [
            {"messages": [{"role": "user", "content": "test"}]},
        ]
        result = dedup.deduplicate(examples)
        assert result.diversity_score == -1.0
        assert result.duplicates_removed == 0
        assert result.deduplicated_count == 1


# =========================================================================
# diversity_score
# =========================================================================


class TestDiversityScore:
    def test_identical_embeddings(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        score = dedup.diversity_score(embeddings)
        assert score == pytest.approx(0.0)

    def test_orthogonal_embeddings(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        score = dedup.diversity_score(embeddings)
        assert score == pytest.approx(1.0)

    def test_single_embedding(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        score = dedup.diversity_score([[1, 0, 0]])
        assert score == 1.0

    def test_empty_embeddings(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        score = dedup.diversity_score([])
        assert score == 1.0

    def test_partially_similar_embeddings(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [0.5, 0.5, 0]]
        score = dedup.diversity_score(embeddings)
        assert 0.0 < score < 1.0

    def test_score_bounded_zero_one(self):
        """Diversity score should always be between 0 and 1."""
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        embeddings = [[1, 0, 0], [-1, 0, 0]]
        score = dedup.diversity_score(embeddings)
        assert 0.0 <= score <= 1.0


# =========================================================================
# _extract_text
# =========================================================================


class TestExtractText:
    def test_messages_format(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        example = {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user question"},
                {"role": "assistant", "content": "assistant answer"},
            ]
        }
        text = dedup._extract_text(example)
        assert "user question" in text
        assert "assistant answer" in text
        assert "system prompt" not in text  # System role excluded

    def test_prompt_response_format(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        example = {"prompt": "hello", "response": "world"}
        text = dedup._extract_text(example)
        assert "hello" in text
        assert "world" in text

    def test_instruction_chosen_format(self):
        """Should handle DPO-style instruction/chosen format."""
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        example = {"instruction": "do something", "chosen": "done"}
        text = dedup._extract_text(example)
        assert "do something" in text
        assert "done" in text

    def test_empty_messages(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        example = {"messages": []}
        text = dedup._extract_text(example)
        assert text == ""

    def test_no_recognized_keys(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        example = {"random_key": "random_value"}
        text = dedup._extract_text(example)
        assert text == ""

    def test_text_capped_per_message(self):
        """Each message content should be capped at 500 chars."""
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        long_content = "x" * 1000
        example = {
            "messages": [
                {"role": "user", "content": long_content},
            ]
        }
        text = dedup._extract_text(example)
        assert len(text) <= 2000

    def test_total_text_capped(self):
        """Total extracted text should be capped at 2000 chars."""
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        example = {
            "messages": [
                {"role": "user", "content": "a" * 500},
                {"role": "assistant", "content": "b" * 500},
                {"role": "user", "content": "c" * 500},
                {"role": "assistant", "content": "d" * 500},
                {"role": "user", "content": "e" * 500},
            ]
        }
        text = dedup._extract_text(example)
        assert len(text) <= 2000


# =========================================================================
# compute_embeddings
# =========================================================================


class TestComputeEmbeddings:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key=None))
        with pytest.raises(RuntimeError, match="NIM API key not configured"):
            dedup.compute_embeddings(["test text"])

    @patch("bashgym.factory.dedup.httpx.Client")
    def test_batching(self, mock_client_class):
        """Should batch texts according to batch_size."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = DedupConfig(nim_api_key="fake", batch_size=2)
        dedup = EmbeddingDeduplicator(config)
        dedup._client = mock_client

        texts = ["text1", "text2", "text3"]
        dedup.compute_embeddings(texts)

        # 3 texts with batch_size=2 -> 2 API calls
        assert mock_client.post.call_count == 2


# =========================================================================
# Context Manager
# =========================================================================


class TestContextManager:
    def test_enter_exit(self):
        with EmbeddingDeduplicator(DedupConfig(nim_api_key="fake")) as dedup:
            assert dedup is not None

    def test_close_cleans_up_client(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        mock_client = MagicMock()
        dedup._client = mock_client
        dedup.close()
        mock_client.close.assert_called_once()
        assert dedup._client is None

    def test_close_no_client(self):
        """Close should be safe when no client exists."""
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        dedup.close()  # Should not raise

    def test_context_manager_calls_close(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        mock_client = MagicMock()
        dedup._client = mock_client

        with dedup:
            pass

        mock_client.close.assert_called_once()


# =========================================================================
# Client property
# =========================================================================


class TestClientProperty:
    def test_lazy_init(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        assert dedup._client is None
        client = dedup.client
        assert client is not None
        assert dedup._client is not None
        dedup.close()

    def test_reuses_client(self):
        dedup = EmbeddingDeduplicator(DedupConfig(nim_api_key="fake"))
        client1 = dedup.client
        client2 = dedup.client
        assert client1 is client2
        dedup.close()
