from bashgym.context_authority import build_context_authority


def test_live_runtime_wins_when_durable_status_disagrees():
    authority = build_context_authority(
        generated_at="2026-07-14T20:00:00+00:00",
        canvas_updated_at="2026-07-14T19:59:00+00:00",
        training_runs=[{"run_id": "run-1", "status": "running"}],
        runtime_jobs=[{"job_id": "run-1", "status": "completed"}],
        campaigns=[],
    )

    assert [item["source_id"] for item in authority["source_precedence"]] == [
        "live_runtime",
        "durable_ledger",
        "workspace_snapshot",
        "curated_gbrain",
        "conversation_memory",
    ]
    assert authority["conflicts"] == [
        {
            "code": "run_status_mismatch",
            "entity_id": "run-1",
            "higher_authority": "live_runtime",
            "higher_value": "completed",
            "lower_authority": "durable_ledger",
            "lower_value": "running",
            "decision_required": False,
            "resolution": "Report runtime status and reconcile the durable run record.",
        }
    ]


def test_empty_runtime_query_is_a_fresh_empty_observation():
    authority = build_context_authority(
        generated_at="2026-07-14T20:00:00+00:00",
        canvas_updated_at=None,
        training_runs=[],
        runtime_jobs=[],
        campaigns=[],
    )

    runtime = authority["sources"][0]
    assert runtime["freshness"] == "fresh"
    assert runtime["empty"] is True
    assert authority["sources"][3]["freshness"] == "unavailable"
    assert authority["sources"][4]["freshness"] == "unverified"
