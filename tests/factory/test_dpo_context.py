import json
from unittest.mock import patch

from bashgym.factory.dpo_pairer import pair_failures_for_dpo


def trace(prompt, passed, context="a" * 64):
    return {
        "trace_id": "chosen" if passed else "rejected",
        "metadata": {
            "user_initial_prompt": prompt,
            "verification_passed": passed,
            "preference_context": {
                "task_id": "repair-1",
                "snapshot_digest": context,
                "tools_digest": "b" * 64,
            },
        },
        "messages": [{"role": "assistant", "content": "fixed" if passed else "broken"}],
    }


def pair(tmp_path, chosen, rejected):
    gold, failed = tmp_path / "gold", tmp_path / "failed"
    gold.mkdir()
    failed.mkdir()
    (gold / "trace.json").write_text(json.dumps(chosen))
    (failed / "trace.json").write_text(json.dumps(rejected))
    with patch("bashgym.factory.dpo_pairer.EmbeddingDeduplicator") as factory:
        factory.return_value.compute_embeddings.side_effect = [[[1.0]], [[1.0]]]
        factory.return_value._cosine_similarity.return_value = 1.0
        return pair_failures_for_dpo(gold, failed)


def test_similarity_does_not_authorize_different_prompt(tmp_path):
    assert pair(tmp_path, trace("fix A", True), trace("fix B", False)) == []


def test_same_prompt_different_snapshot_is_ineligible(tmp_path):
    assert pair(tmp_path, trace("fix", True), trace("fix", False, "c" * 64)) == []


def test_unverified_gold_is_ineligible(tmp_path):
    chosen = trace("fix", True)
    chosen["metadata"].pop("verification_passed")
    assert pair(tmp_path, chosen, trace("fix", False)) == []


def test_legacy_trace_preserved_but_ineligible(tmp_path):
    chosen = trace("fix", True)
    chosen["metadata"].pop("preference_context")
    assert pair(tmp_path, chosen, trace("fix", False)) == []
    assert (tmp_path / "gold" / "trace.json").exists()


def test_verified_same_context_pair_has_binding(tmp_path):
    pairs = pair(tmp_path, trace("fix", True), trace("fix", False))
    assert len(pairs) == 1
    assert pairs[0].metadata["conditioning_verified"] is True
    assert len(pairs[0].metadata["conditioning_digest"]) == 64
