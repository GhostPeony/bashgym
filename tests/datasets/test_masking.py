"""Tests for the SFT loss-masking policy."""

from bashgym.datasets.masking import observations_masked, trainable_message_mask

CONVO = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do X"},
    {
        "role": "assistant",
        "content": "running tool",
        "tool_calls": [{"function": {"name": "Bash"}}],
    },
    {"role": "tool", "content": "output"},
    {"role": "assistant", "content": "done"},
]


class TestTrainableMessageMask:
    def test_only_assistant_turns_trainable(self):
        assert trainable_message_mask(CONVO) == [False, False, True, False, True]

    def test_empty(self):
        assert trainable_message_mask([]) == []


class TestObservationsMasked:
    def test_valid_conversation(self):
        assert observations_masked(CONVO) is True

    def test_no_assistant_turn_is_invalid(self):
        assert observations_masked([{"role": "user", "content": "hi"}]) is False

    def test_tool_only_is_invalid(self):
        assert observations_masked([{"role": "tool", "content": "out"}]) is False
