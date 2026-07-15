from __future__ import annotations

from scripts.memexai import generate_dd_train_pairs as gen


def _corpus_rows() -> list[dict[str, object]]:
    long_text = "The guest explains retrieval eval design and why local windows matter. " * 8
    return [
        {
            "chunk_id": f"chunk-{video}-{idx}",
            "youtube_video_id": f"video-{video}",
            "chunk_index": idx,
            "title": f"Video {video}",
            "channel": "Channel A",
            "url": f"https://example.test/v{video}",
            "text": long_text,
        }
        for video in (1, 2, 3)
        for idx in range(4)
    ]


def _template_rows() -> list[dict[str, object]]:
    return [
        {"split": "train", "positive_video_id": "video-1", "positive_chunk_id": "chunk-1-0"},
        {"split": "train", "positive_video_id": "video-2", "positive_chunk_id": "chunk-2-1"},
        {"split": "test", "positive_video_id": "video-3", "positive_chunk_id": "chunk-3-0"},
    ]


def _fake_dd_row(seed_index: int = 0, transcript_words: int = 200) -> dict[str, object]:
    transcript = " ".join(
        f"word{idx} speculative decoding lets the small model draft tokens"
        for idx in range(0, transcript_words, 9)
    )
    return {
        "seed_index": seed_index,
        "seed_angle": "a concrete technical deep-dive",
        "seed_style_title": "Video 1",
        "seed_style_channel": "Channel A",
        "seed_style_video_id": "video-1",
        "seed_style_chunk_id": "chunk-1-0",
        "seed_style_text": "Completely different style text about reward models.",
        "synthetic_transcript": transcript,
        "query_variants": {
            "natural_question": "How does speculative decoding use a draft model?",
            "keyword_query": "speculative decoding draft model",
            "semantic_paraphrase": "Find the part on accelerating inference with a smaller model.",
        },
        "query_judgement": {
            "natural_question_pass": True,
            "natural_question_reason": "Grounded.",
            "keyword_query_pass": True,
            "keyword_query_reason": "Grounded.",
            "semantic_paraphrase_pass": False,
            "semantic_paraphrase_reason": "Too vague.",
        },
    }


def test_train_video_ids_and_used_positives_come_from_template_splits() -> None:
    rows = _template_rows()

    assert gen.train_video_ids(rows) == {"video-1", "video-2"}
    assert gen.used_positive_chunk_ids(rows) == {"chunk-1-0", "chunk-2-1", "chunk-3-0"}


def test_select_real_seed_chunks_excludes_used_and_non_train_videos() -> None:
    seeds = gen.select_real_seed_chunks(
        _corpus_rows(),
        train_videos={"video-1", "video-2"},
        used_chunk_ids={"chunk-1-0", "chunk-2-1"},
        num_seeds=100,
        min_chunk_chars=300,
        seed=7,
    )

    chunk_ids = {row["chunk_id"] for row in seeds}
    assert "chunk-1-0" not in chunk_ids
    assert "chunk-2-1" not in chunk_ids
    assert all(not chunk_id.startswith("chunk-3") for chunk_id in chunk_ids)
    assert len(seeds) == 6


def test_select_real_seed_chunks_is_deterministic() -> None:
    kwargs = dict(
        train_videos={"video-1", "video-2"},
        used_chunk_ids=set(),
        num_seeds=4,
        min_chunk_chars=300,
        seed=11,
    )

    first = gen.select_real_seed_chunks(_corpus_rows(), **kwargs)
    second = gen.select_real_seed_chunks(_corpus_rows(), **kwargs)

    assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]


def test_fake_seed_rows_cycle_angles_and_exemplars() -> None:
    exemplars = [row for row in _corpus_rows() if row["youtube_video_id"] == "video-1"][:2]

    rows = gen.fake_seed_rows(exemplars, num_seeds=10, style_excerpt_chars=120)

    assert len(rows) == 10
    assert rows[0]["seed_angle"] == gen.SYNTHETIC_ANGLES[0]
    assert rows[8]["seed_angle"] == gen.SYNTHETIC_ANGLES[0]
    assert rows[0]["seed_style_chunk_id"] != rows[1]["seed_style_chunk_id"]
    assert all(len(str(row["seed_style_text"])) <= 120 for row in rows)


def test_synthetic_chunk_validation_rejects_slop_and_markdown() -> None:
    good = "the model just drafts a few tokens ahead and then you verify them " * 12

    assert gen.synthetic_chunk_is_valid(good) == (True, None)
    assert gen.synthetic_chunk_is_valid("too short") == (False, "too_short")
    assert gen.synthetic_chunk_is_valid("As an AI, I cannot " + good)[1] == "refusal_or_ai_slop"
    assert gen.synthetic_chunk_is_valid(good + "\n- bullet point")[1] == "markdown_formatting"
    style = "identical style exemplar text " * 20
    assert gen.synthetic_chunk_is_valid(style + good, style)[1] == "copied_style_exemplar"


def test_flatten_fake_rows_builds_trainer_compatible_rows() -> None:
    query_rows, chunk_rows, failures = gen.flatten_fake_rows(
        [_fake_dd_row()], "local-model", require_judge=True
    )

    assert failures["semantic_paraphrase:judge_rejected"] == 1
    assert len(chunk_rows) == 1
    chunk = chunk_rows[0]
    assert chunk["chunk_id"].startswith("synthetic-")
    assert chunk["youtube_video_id"] == "synthetic-video-000"
    assert chunk["channel"] == gen.SYNTHETIC_CHANNEL
    assert [row["query_type"] for row in query_rows] == ["natural_question", "keyword_query"]
    for row in query_rows:
        assert row["split"] == "train"
        assert row["positive_chunk_id"] == chunk["chunk_id"]
        assert row["positive_video_id"] == chunk["youtube_video_id"]


def test_flatten_fake_rows_drops_invalid_synthetic_chunks() -> None:
    bad = _fake_dd_row()
    bad["synthetic_transcript"] = "way too short to be a transcript"

    query_rows, chunk_rows, failures = gen.flatten_fake_rows(
        [bad], "local-model", require_judge=True
    )

    assert query_rows == []
    assert chunk_rows == []
    assert failures["synthetic_chunk:too_short"] == 1


def test_dedupe_query_rows_drops_repeated_query_text() -> None:
    rows = [
        {"query": "What is speculative decoding?"},
        {"query": "what is  speculative decoding?"},
        {"query": "How do draft models work?"},
    ]

    kept, dropped = gen.dedupe_query_rows(rows)

    assert [row["query"] for row in kept] == [
        "What is speculative decoding?",
        "How do draft models work?",
    ]
    assert dropped == 1
