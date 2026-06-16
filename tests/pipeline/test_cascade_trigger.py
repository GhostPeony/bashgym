"""Tests for the gold-count cascade auto-trigger (ThresholdMonitor.should_cascade)."""

import json
from unittest import mock

from bashgym.pipeline.config import PipelineConfig
from bashgym.pipeline.orchestrator import Pipeline
from bashgym.pipeline.threshold_monitor import ThresholdMonitor
from bashgym.trace_capture.importers.claude_history import ImportResult


def _gold(tmp_path, n):
    d = tmp_path / "gold"
    d.mkdir(exist_ok=True)
    for i in range(n):
        (d / f"t{i}.json").write_text("{}")
    return d


class TestShouldCascade:
    def test_below_threshold(self, tmp_path):
        g = _gold(tmp_path, 5)
        m = ThresholdMonitor(PipelineConfig(cascade_enabled=True, cascade_gold_threshold=10))
        assert m.should_cascade(g) is False

    def test_at_threshold(self, tmp_path):
        g = _gold(tmp_path, 10)
        m = ThresholdMonitor(PipelineConfig(cascade_enabled=True, cascade_gold_threshold=10))
        assert m.should_cascade(g) is True

    def test_disabled_by_default(self, tmp_path):
        g = _gold(tmp_path, 50)
        m = ThresholdMonitor(PipelineConfig())  # cascade_enabled defaults False
        assert m.should_cascade(g) is False

    def test_watermark_prevents_retrigger(self, tmp_path):
        g = _gold(tmp_path, 10)
        wm = tmp_path / "wm.json"
        m = ThresholdMonitor(
            PipelineConfig(cascade_enabled=True, cascade_gold_threshold=10), watermark_path=wm
        )
        assert m.should_cascade(g) is True
        m.mark_cascade_triggered(g)
        assert m.should_cascade(g) is False  # needs 10 MORE gold than the watermark
        for i in range(10):
            (g / f"more{i}.json").write_text("{}")
        assert m.should_cascade(g) is True

    def test_config_field_roundtrips(self):
        cfg = PipelineConfig(cascade_enabled=True, cascade_gold_threshold=300)
        restored = PipelineConfig.from_dict(cfg.to_dict())
        assert restored.cascade_enabled is True and restored.cascade_gold_threshold == 300


def _isolated_pipeline(tmp_path, cfg, on_event=None):
    """Build a Pipeline whose trace dirs are isolated tmp dirs and importer is mockable."""
    gold = tmp_path / "gold"
    failed = tmp_path / "failed"
    pending = tmp_path / "pending"
    for d in (gold, failed, pending):
        d.mkdir(exist_ok=True)
    pipeline = Pipeline(
        config_path=tmp_path / "config.json",
        bashgym_dir=tmp_path / "bashgym",
        on_event=on_event,
    )
    pipeline.config = cfg
    pipeline._gate.config = cfg
    pipeline._monitor.config = cfg
    pipeline._trace_capture.gold_traces_dir = gold
    pipeline._trace_capture.failed_traces_dir = failed
    pipeline._trace_capture.traces_dir = pending
    return pipeline, gold


def _gold_trace_dest(tmp_path, name):
    dest = tmp_path / name
    dest.write_text(json.dumps({"summary": {"success_rate": 1.0, "total_steps": 5}}))
    return dest


class TestOrchestratorCascadeWiring:
    def test_cascade_trigger_fires_and_emits(self, tmp_path):
        events = []
        cfg = PipelineConfig(
            classify_gold_min_steps=1,
            classify_gold_min_success_rate=0.5,
            cascade_enabled=True,
            cascade_gold_threshold=1,
        )
        pipeline, gold = _isolated_pipeline(
            tmp_path, cfg, on_event=lambda t, p: events.append((t, p))
        )
        for i in range(3):  # pre-fill above threshold
            (gold / f"g{i}.json").write_text("{}")

        triggered = []
        pipeline.cascade_trigger = lambda n: triggered.append(n)

        dest = _gold_trace_dest(tmp_path, "trace_s1.json")
        fake = ImportResult(
            session_id="s1",
            source_file=tmp_path / "src.jsonl",
            steps_imported=2,
            destination_file=dest,
        )
        with mock.patch.object(pipeline._importer, "import_session", return_value=fake):
            result = pipeline.handle_session_file(tmp_path / "src.jsonl")

        assert result is not None
        cascade_evts = [
            p
            for t, p in events
            if t == "pipeline:threshold_reached" and p.get("stage") == "cascade"
        ]
        assert len(cascade_evts) == 1
        assert cascade_evts[0]["gold_count"] >= 1
        assert len(triggered) == 1  # callback received the gold count

    def test_cascade_disabled_no_trigger(self, tmp_path):
        events = []
        cfg = PipelineConfig(
            classify_gold_min_steps=1, classify_gold_min_success_rate=0.5
        )  # cascade_enabled defaults False
        pipeline, gold = _isolated_pipeline(
            tmp_path, cfg, on_event=lambda t, p: events.append((t, p))
        )
        for i in range(5):
            (gold / f"g{i}.json").write_text("{}")

        triggered = []
        pipeline.cascade_trigger = lambda n: triggered.append(n)

        dest = _gold_trace_dest(tmp_path, "trace_s2.json")
        fake = ImportResult(
            session_id="s2",
            source_file=tmp_path / "src.jsonl",
            steps_imported=2,
            destination_file=dest,
        )
        with mock.patch.object(pipeline._importer, "import_session", return_value=fake):
            pipeline.handle_session_file(tmp_path / "src.jsonl")

        assert triggered == []
        assert not [p for t, p in events if p.get("stage") == "cascade"]

    def test_cascade_callback_error_is_isolated(self, tmp_path):
        """A throwing cascade_trigger must not break the pipeline; it emits cascade_error."""
        events = []
        cfg = PipelineConfig(
            classify_gold_min_steps=1,
            classify_gold_min_success_rate=0.5,
            cascade_enabled=True,
            cascade_gold_threshold=1,
        )
        pipeline, gold = _isolated_pipeline(
            tmp_path, cfg, on_event=lambda t, p: events.append((t, p))
        )
        (gold / "g0.json").write_text("{}")

        def boom(_n):
            raise RuntimeError("cascade boom")

        pipeline.cascade_trigger = boom

        dest = _gold_trace_dest(tmp_path, "trace_s3.json")
        fake = ImportResult(
            session_id="s3",
            source_file=tmp_path / "src.jsonl",
            steps_imported=2,
            destination_file=dest,
        )
        with mock.patch.object(pipeline._importer, "import_session", return_value=fake):
            result = pipeline.handle_session_file(tmp_path / "src.jsonl")

        assert result is not None  # pipeline still returns a result despite the callback error
        err_evts = [p for t, p in events if t == "pipeline:cascade_error"]
        assert len(err_evts) == 1 and "cascade boom" in err_evts[0]["error"]
