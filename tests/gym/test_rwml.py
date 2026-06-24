"""Tests for the RWML (Reinforcement World Model Learning) reward layer.

RWML (arXiv:2602.05842) trains an action-conditioned world model as a
self-supervised pre-RL stage: the agent predicts the next environment state
from history+action, rewarded by an embedding-space binary signal
    r_WM = 1[ 1 - cos(E(s_hat), E(s)) < tau_d ]
optimized with GRPO. This module is the pure, embedding-provider-agnostic
contract (an ``embed_fn`` is injected) that a terminal-RL backend consumes.
"""

import pytest

from bashgym.gym.rwml import (
    RWML_DEFAULT_DISTANCE_THRESHOLD,
    RWMLConfig,
    WorldModelTransition,
    build_world_model_reward_fn,
    cosine_similarity,
    extract_transitions,
    group_relative_advantages,
    keep_easy_sample,
    world_model_reward,
)


def test_rwml_config_defaults():
    config = RWMLConfig()

    assert config.enabled is False
    assert config.distance_threshold == RWML_DEFAULT_DISTANCE_THRESHOLD
    assert config.easy_keep_probability == 0.1
    assert config.history_window == 4
    snapshot = config.to_dict()
    assert snapshot["enabled"] is False
    assert snapshot["distance_threshold"] == RWML_DEFAULT_DISTANCE_THRESHOLD
    assert snapshot["embedding_model"] == ""


def test_cosine_similarity_orthogonal_and_identical():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_world_model_reward_is_binary_around_threshold():
    vectors = {
        "exact": [1.0, 0.0],
        "close": [0.99, 0.14],  # ~0.99 cos sim -> distance ~0.01
        "far": [0.0, 1.0],  # distance 1.0
    }

    def embed(text):
        return vectors[text]

    # identical prediction -> distance 0 < tau -> reward 1.0
    assert world_model_reward("exact", "exact", embed, distance_threshold=0.2) == 1.0
    # near prediction inside the trust threshold -> reward 1.0
    assert world_model_reward("close", "exact", embed, distance_threshold=0.2) == 1.0
    # far prediction -> reward 0.0
    assert world_model_reward("far", "exact", embed, distance_threshold=0.2) == 0.0


def test_extract_transitions_builds_history_windowed_triplets():
    transitions = extract_transitions(
        [
            {"action": "ls", "state": "fileA fileB"},
            {"action": "cat fileA", "state": "hello"},
            {"action": "rm fileA", "state": ""},
        ],
        instruction="clean up the workspace",
        history_window=1,
    )

    assert transitions == [
        WorldModelTransition(
            instruction="clean up the workspace",
            prior=(),
            action="ls",
            next_state="fileA fileB",
        ),
        WorldModelTransition(
            instruction="clean up the workspace",
            prior=(("ls", "fileA fileB"),),
            action="cat fileA",
            next_state="hello",
        ),
        WorldModelTransition(
            instruction="clean up the workspace",
            prior=(("cat fileA", "hello"),),
            action="rm fileA",
            next_state="",
        ),
    ]


def test_extract_transitions_respects_larger_history_window():
    transitions = extract_transitions(
        [
            {"action": "a", "state": "s1"},
            {"action": "b", "state": "s2"},
            {"action": "c", "state": "s3"},
        ],
        history_window=4,
    )

    assert transitions[2].prior == (("a", "s1"), ("b", "s2"))
    assert transitions[2].action == "c"


def test_group_relative_advantages_standardizes_rewards():
    advantages = group_relative_advantages([1.0, 0.0, 1.0, 0.0])

    assert advantages == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_group_relative_advantages_zero_variance_returns_zeros():
    assert group_relative_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert group_relative_advantages([]) == []


def test_keep_easy_sample_drops_most_easy_keeps_all_hard():
    # easy sample (high pass rate): kept only when the draw is below p
    assert keep_easy_sample(0.95, threshold=0.8, keep_probability=0.1, draw=0.05) is True
    assert keep_easy_sample(0.95, threshold=0.8, keep_probability=0.1, draw=0.5) is False
    # hard sample (low pass rate): always kept regardless of the draw
    assert keep_easy_sample(0.4, threshold=0.8, keep_probability=0.1, draw=0.99) is True


def test_build_world_model_reward_fn_scores_a_batch_of_predictions():
    vectors = {"hit": [1.0, 0.0], "miss": [0.0, 1.0]}

    def embed(text):
        return vectors[text]

    reward_fn = build_world_model_reward_fn(embed, distance_threshold=0.2)

    # predicted vs actual next states; exact match -> 1.0, orthogonal -> 0.0
    assert reward_fn(["hit", "miss"], ["hit", "hit"]) == [1.0, 0.0]
