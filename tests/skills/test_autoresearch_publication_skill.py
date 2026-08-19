import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SKILL = ROOT / "assistant" / "workspace" / "skills" / "autoresearch-publication"


def _valid_post(*, status: str = "draft") -> dict:
    approval = {
        "status": status,
        "approved_by": None,
        "approved_at": None,
        "feedback": [],
    }
    if status == "approved":
        approval.update(
            approved_by="human-reviewer",
            approved_at="2026-08-15T20:00:00Z",
        )
    return {
        "schema_version": "open_frontiers.research_post.v1",
        "publication": {
            "slug": "fixed-eval-example",
            "title": "A fixed-evaluation model experiment",
            "summary": "One controlled training change improved the fixed primary metric.",
            "approval": approval,
        },
        "experiment": {
            "question": "Does the declared intervention improve the fixed evaluation?",
            "hypothesis": "The intervention will improve exact accuracy.",
            "model": "Example model",
            "method": "LoRA supervised fine-tuning",
            "intervention": "Increase the approved training step count.",
            "evaluation": "Unchanged 64-example held-out suite.",
            "method_selection": {
                "selected_method": "lora_sft",
                "selection_authority": "host_agent",
                "rationale": "Demonstrations were available and the failure was supervised behavior.",
                "alternatives": [
                    {
                        "method": "grpo",
                        "status": "not_selected",
                        "reason": "No rollout-readiness evidence was collected for this experiment.",
                    }
                ],
            },
        },
        "results": {
            "primary": {
                "name": "exact_accuracy",
                "unit": "fraction",
                "baseline": 0.1,
                "candidate": 0.7,
                "delta": 0.6,
                "direction": "higher_is_better",
            },
            "secondary": [],
            "decision": "keep",
            "failure_analysis": [
                {
                    "category": "count_error",
                    "summary": "Responses with incorrect object counts decreased.",
                    "baseline_count": 24,
                    "candidate_count": 6,
                    "delta": -18,
                    "status": "improved",
                }
            ],
        },
        "narrative": {
            "simple": "The trained candidate answered more held-out examples correctly.",
            "technical": "The candidate improved exact accuracy by 0.60 under the unchanged evaluator.",
            "judgement": "Keep the candidate for the next controlled comparison.",
            "limitations": ["The held-out suite is small and task-specific."],
            "next_experiment": "Test a related unseen slice without changing the evaluator.",
        },
        "training_rungs": [
            {
                "order": 1,
                "label": "Baseline",
                "method": "Fixed evaluation",
                "status": "completed",
                "summary": "Measure the unchanged model.",
            },
            {
                "order": 2,
                "label": "Candidate",
                "method": "LoRA SFT",
                "status": "kept",
                "summary": "Train and evaluate one controlled candidate.",
            },
        ],
        "visuals": [
            {
                "id": "primary-comparison",
                "type": "metric_comparison",
                "title": "Exact accuracy",
                "data": {"baseline": 0.1, "candidate": 0.7},
            },
            {
                "id": "training-rungs",
                "type": "training_rungs",
                "title": "Experiment sequence",
                "data": {"orders": [1, 2]},
            },
        ],
        "claims": [
            {
                "id": "primary-improvement",
                "text": "Exact accuracy improved by 0.60.",
                "evidence_refs": ["campaign-export"],
                "source_refs": [],
            }
        ],
        "sources": [
            {
                "id": "method-source",
                "title": "A primary research source",
                "url": "https://arxiv.org/abs/2502.14499",
                "year": 2025,
                "role": "method_context",
            }
        ],
        "provenance": {
            "evidence": [
                {
                    "id": "campaign-export",
                    "kind": "bashgym_campaign_export",
                    "digest": "a" * 64,
                }
            ],
            "generated_at": "2026-08-15T20:00:00Z",
        },
    }


def _run(tmp_path: Path, post: dict, *args: str) -> subprocess.CompletedProcess[str]:
    post_path = tmp_path / "post.json"
    post_path.write_text(json.dumps(post), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "research_post.py"), *args, str(post_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_skill_defines_renderer_neutral_human_approved_output_contract():
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    contract = (SKILL / "references" / "output-contract.md").read_text(encoding="utf-8")
    template = json.loads(
        (SKILL / "assets" / "research-post.template.json").read_text(encoding="utf-8")
    )

    assert "human-approved" in skill
    assert "campaign_evidence.json" in skill
    assert "research_post.py validate" in skill
    assert "research_post.py render" in skill
    assert "Do not generate HTML" in skill
    assert "open_frontiers.research_post.v1" in contract
    assert "simple" in contract and "technical" in contract
    assert "training_rungs" in contract
    assert "metric_comparison" in contract
    assert "training_rungs" in contract
    assert template["schema_version"] == "open_frontiers.research_post.v1"
    assert template["publication"]["approval"]["status"] == "draft"


def test_validator_accepts_draft_and_human_approved_packages(tmp_path: Path):
    draft = _valid_post()
    draft["publication"]["approval"]["feedback"] = [
        {"note": "Clarify the task-specific limitation.", "status": "open"}
    ]
    assert _run(tmp_path, draft, "validate").returncode == 0
    assert _run(tmp_path, _valid_post(status="approved"), "validate").returncode == 0


def test_validator_rejects_inconsistent_metrics_and_false_approval(tmp_path: Path):
    bad_delta = _valid_post()
    bad_delta["results"]["primary"]["delta"] = 0.5
    result = _run(tmp_path, bad_delta, "validate")
    assert result.returncode == 2
    assert "primary delta" in result.stderr

    false_approval = _valid_post(status="approved")
    false_approval["publication"]["approval"]["approved_by"] = None
    result = _run(tmp_path, false_approval, "validate")
    assert result.returncode == 2
    assert "human approval" in result.stderr

    unresolved = _valid_post(status="approved")
    unresolved["publication"]["approval"]["feedback"] = [
        {"note": "Add uncertainty context.", "status": "open"}
    ]
    result = _run(tmp_path, unresolved, "validate")
    assert result.returncode == 2
    assert "unresolved feedback" in result.stderr


def test_validator_rejects_missing_dual_context_and_private_material(tmp_path: Path):
    missing_context = _valid_post()
    missing_context["narrative"]["technical"] = ""
    result = _run(tmp_path, missing_context, "validate")
    assert result.returncode == 2
    assert "narrative.technical" in result.stderr

    private_material = _valid_post()
    private_material["narrative"]["simple"] = "Read C:\\Users\\operator\\secret.json"
    result = _run(tmp_path, private_material, "validate")
    assert result.returncode == 2
    assert "private path" in result.stderr


def test_validator_rejects_broken_references_and_nonfinite_metrics(tmp_path: Path):
    broken = _valid_post()
    broken["claims"][0]["evidence_refs"] = ["missing"]
    result = _run(tmp_path, broken, "validate")
    assert result.returncode == 2
    assert "unknown evidence" in result.stderr

    nonfinite = _valid_post()
    nonfinite["results"]["primary"]["candidate"] = float("nan")
    result = _run(tmp_path, nonfinite, "validate")
    assert result.returncode == 2
    assert "finite" in result.stderr


def test_renderer_produces_review_markdown_from_valid_json(tmp_path: Path):
    output = tmp_path / "post.md"
    post = _valid_post()
    post["publication"]["approval"]["feedback"] = [
        {"note": "Clarify the task-specific limitation.", "status": "open"}
    ]
    result = _run(tmp_path, post, "render", "--output", str(output))

    assert result.returncode == 0, result.stderr
    rendered = output.read_text(encoding="utf-8")
    assert "# A fixed-evaluation model experiment" in rendered
    assert "## In plain language" in rendered
    assert "## Technical interpretation" in rendered
    assert "## Training rungs" in rendered
    assert "## Method selection" in rendered
    assert "Demonstrations were available" in rendered
    assert "## Failure analysis" in rendered
    assert "count_error" in rendered
    assert "## Review feedback" in rendered
    assert "[open] Clarify the task-specific limitation." in rendered
    assert "Draft — human approval required" in rendered


def test_validator_rejects_raw_failure_examples_and_inconsistent_counts(tmp_path: Path):
    raw_example = _valid_post()
    raw_example["results"]["failure_analysis"][0]["prediction"] = "raw model output"
    result = _run(tmp_path, raw_example, "validate")
    assert result.returncode == 2
    assert "unexpected fields" in result.stderr

    bad_delta = _valid_post()
    bad_delta["results"]["failure_analysis"][0]["delta"] = -17
    result = _run(tmp_path, bad_delta, "validate")
    assert result.returncode == 2
    assert "failure delta" in result.stderr
