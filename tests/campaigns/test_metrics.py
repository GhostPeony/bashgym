"""Training metric normalization, diagnostics, persistence, and projection tests."""

from datetime import timedelta

import pytest

from bashgym.campaigns.metrics import (
    AlertSeverity,
    MetricStreamError,
    detect_training_alerts,
    parse_metric_lines,
    trackio_projection,
)
from bashgym.campaigns.persistence import CampaignPersistenceError
from tests.campaigns.test_remote_persistence import _claimed_attempt
from tests.campaigns.test_worker import START


def test_metric_jsonl_is_normalized_and_projects_to_trackio_shape():
    points = parse_metric_lines(
        (
            '{"step":2,"loss":0.25,"learning_rate":0.0001,"label":"ignored"}',
            '{"global_step":1,"loss":0.5}',
        )
    )
    assert points[0].values == {"learning_rate": 0.0001, "loss": 0.25}
    assert trackio_projection(
        project="memexai", run="attempt-1", metric_name="loss", points=points
    ) == {
        "project": "memexai",
        "run": "attempt-1",
        "metric": "loss",
        "values": [{"step": 1, "value": 0.5}, {"step": 2, "value": 0.25}],
    }


@pytest.mark.parametrize(
    "line",
    [
        "not-json",
        '{"loss":0.5}',
        '{"step":-1,"loss":0.5}',
        '{"step":1,"loss":NaN}',
        '{"step":1,"label":"no numeric metric"}',
    ],
)
def test_metric_stream_rejects_malformed_or_non_finite_evidence(line):
    with pytest.raises(MetricStreamError):
        parse_metric_lines((line,))


def test_diagnostics_flag_loss_spikes_and_near_zero():
    points = parse_metric_lines(
        (
            '{"step":1,"loss":0.2}',
            '{"step":2,"loss":0.8}',
            '{"step":3,"loss":0.000000001}',
        )
    )
    alerts = detect_training_alerts(points)
    assert [(alert.code, alert.severity) for alert in alerts] == [
        ("loss_spike", AlertSeverity.ERROR),
        ("loss_near_zero", AlertSeverity.WARN),
    ]


def test_metric_persistence_is_idempotent_ordered_and_conflict_safe(tmp_path):
    repository, attempt = _claimed_attempt(tmp_path)
    lines = ('{"step":2,"loss":0.25}', '{"step":1,"loss":0.5}')
    first = repository.append_remote_metrics(
        attempt,
        lines,
        source="training_metrics.jsonl",
        cursor_end=64,
        now=START + timedelta(seconds=2),
    )
    replay = repository.append_remote_metrics(
        attempt,
        lines,
        source="training_metrics.jsonl",
        cursor_end=64,
        now=START + timedelta(seconds=3),
    )
    assert replay == first
    series = repository.get_metric_series("workspace-a", attempt.attempt_id, "loss")
    assert [(point.step, point.value) for point in series] == [(1, 0.5), (2, 0.25)]
    events = repository.list_events("workspace-a", "campaign-1")
    assert sum(
        event.event_type == "campaign:training-metrics-appended" for _, event in events
    ) == 1

    with pytest.raises(CampaignPersistenceError, match="metric_value_conflict"):
        repository.append_remote_metrics(
            attempt,
            ('{"step":2,"loss":0.75}',),
            source="training_metrics.jsonl",
            cursor_end=80,
            now=START + timedelta(seconds=4),
        )
