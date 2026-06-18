"""Tests for recording held-out eval verdicts into the model registry (S3 registry)."""

from bashgym.eval.heldout import evaluate_candidate, first_gold_tool_call
from bashgym.models.profile import ModelProfile
from bashgym.models.registry import ModelRegistry


def _ex(session, name="Bash", args=None):
    args = {"command": "ls"} if args is None else args
    return {
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
            },
        ],
        "metadata": {"session_id": session},
    }


def _wrong(ex):
    return {"type": "function", "function": {"name": "WrongTool", "arguments": {}}}


def _ship_report_dict():
    """A real HeldoutReport.to_dict() where the candidate beats the base (ships)."""
    examples = [_ex(f"s{i % 3}") for i in range(9)]
    return evaluate_candidate(examples, _wrong, first_gold_tool_call, seed=0).to_dict()


class TestProfileHeldout:
    def test_add_and_latest(self):
        p = ModelProfile(model_id="m1", run_id="r1", display_name="M1")
        assert p.latest_heldout_eval is None
        p.add_heldout_eval({"ship": True, "trace_delta": 1.0})
        p.add_heldout_eval({"ship": False, "trace_delta": 0.0})
        assert len(p.heldout_evals) == 2
        assert p.latest_heldout_eval["ship"] is False  # newest last

    def test_keep_trims_history(self):
        p = ModelProfile(model_id="m1", run_id="r1", display_name="M1")
        for i in range(25):
            p.add_heldout_eval({"i": i}, keep=20)
        assert len(p.heldout_evals) == 20
        assert p.heldout_evals[0]["i"] == 5  # oldest 5 trimmed
        assert p.latest_heldout_eval["i"] == 24

    def test_roundtrip_persists_heldout(self, tmp_path):
        p = ModelProfile(
            model_id="m1", run_id="r1", display_name="M1", model_dir=str(tmp_path / "m1")
        )
        p.add_heldout_eval({"ship": True, "trace_delta": 0.4})
        path = p.save()
        loaded = ModelProfile.load(path)
        assert loaded.latest_heldout_eval == {"ship": True, "trace_delta": 0.4}


class TestRegistryRecord:
    def test_record_heldout_eval(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        profile = ModelProfile(
            model_id="m1", run_id="r1", display_name="M1", model_dir=str(tmp_path / "m1")
        )
        reg._profiles["m1"] = profile

        report = _ship_report_dict()
        assert report["ship"] is True

        out = reg.record_heldout_eval("m1", report)
        assert out is not None
        assert out.latest_heldout_eval["ship"] is True

        # persisted to disk and reloads
        reloaded = ModelProfile.load(tmp_path / "m1" / "model_profile.json")
        assert reloaded.latest_heldout_eval["trace_delta"] == report["trace_delta"]

    def test_record_unknown_model_returns_none(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        assert reg.record_heldout_eval("nope", {"ship": True}) is None
