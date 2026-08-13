import re

from fastapi.testclient import TestClient

from bashgym.api import agent_routes, skill_lab_routes
from bashgym.api.routes import app


def _client() -> TestClient:
    return TestClient(app)


def _make_skill(monkeypatch, tmp_path, content: str = "# skill\n") -> str:
    root = tmp_path / "skills"
    skill_dir = root / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    monkeypatch.setattr(agent_routes, "_toolkit_skill_root_candidates", lambda: [("test", root)])
    skills = agent_routes._scan_skill_roots()[1]
    return next(skill.skill_id for skill in skills if skill.source == "test")


def test_contract_persists_by_workspace_and_run_returns_kpis(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    skill_id = _make_skill(
        monkeypatch,
        tmp_path,
        "---\nname: demo-skill\ndescription: Demo\nallowed_tools: [Read]\n---\nUse demo.\n",
    )

    contract = _client().put(
        f"/api/skill-lab/contracts/{skill_id}",
        params={"workspace_id": "workspace-a"},
        json={
            "cases": [
                {
                    "id": "positive",
                    "prompt": "finish the task",
                    "expected_patterns": ["DONE"],
                    "should_invoke": True,
                }
            ],
            "thresholds": {
                "min_uplift": 0,
                "min_routing_precision": 1,
                "min_routing_recall": 1,
            },
        },
    )
    assert contract.status_code == 200
    assert contract.json()["skill_id"] == skill_id
    assert (
        _client()
        .get(f"/api/skill-lab/contracts/{skill_id}", params={"workspace_id": "workspace-b"})
        .status_code
        == 404
    )

    async def fake_chat(endpoint_id, request):
        arm = re.search(r"Skill Lab arm: (\w+)", request.message).group(1)
        return {
            "response": f"DONE\nSKILL_LAB_SHOULD_INVOKE: {'false' if arm == 'baseline' else 'true'}",
            "model": "test-model",
        }

    monkeypatch.setattr(agent_routes, "chat_with_agent_endpoint", fake_chat)
    launched = _client().post(
        "/api/skill-lab/runs",
        json={
            "workspace_id": "workspace-a",
            "skill_id": skill_id,
            "endpoint_id": "hermes",
            "cases": [
                {
                    "case_id": "positive",
                    "prompt": "finish the task",
                    "expected_patterns": ["DONE"],
                    "should_invoke": True,
                }
            ],
            "thresholds": {
                "min_uplift": 0,
                "min_routing_precision": 1,
                "min_routing_recall": 1,
            },
        },
    )
    assert launched.status_code == 202
    run_id = launched.json()["run_id"]

    run = _client().get(f"/api/skill-lab/runs/{run_id}")
    assert run.status_code == 200
    data = run.json()
    assert data["status"] == "completed"
    assert data["model"] == "test-model"
    assert data["progress"] == {"completed": 3, "total": 3}
    assert data["kpis"]["routing_precision"] == 1
    assert data["kpis"]["routing_recall"] == 1
    assert data["kpis"]["routing_f1"] == 1
    assert data["kpis"]["sample_counts"]["attempts"] == 3
    assert data["kpis"]["verdict"] == "effective"
    assert len(data["attempts"]) == 3

    listed_a = _client().get("/api/skill-lab/runs", params={"workspace_id": "workspace-a"})
    listed_b = _client().get("/api/skill-lab/runs", params={"workspace_id": "workspace-b"})
    assert listed_a.status_code == 200
    assert [item["run_id"] for item in listed_a.json()] == [run_id]
    assert listed_b.json() == []
    assert list((tmp_path / "state" / "skill_lab" / "suites").rglob("*.json"))


def test_run_kpi_uplift_and_fail_closed_marker(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    skill_id = _make_skill(monkeypatch, tmp_path)
    responses_without_marker = False

    async def fake_chat(endpoint_id, request):
        nonlocal responses_without_marker
        arm = re.search(r"Skill Lab arm: (\w+)", request.message).group(1)
        if responses_without_marker:
            return {"response": "DONE"}
        positive_case = "\none\n\n" in request.message
        has_done = not positive_case or arm != "baseline"
        invokes = positive_case if arm == "available" else arm == "forced"
        return {
            "response": (
                ("DONE" if has_done else "NOPE")
                + f"\nSKILL_LAB_SHOULD_INVOKE: {'true' if invokes else 'false'}"
            )
        }

    monkeypatch.setattr(agent_routes, "chat_with_agent_endpoint", fake_chat)
    response = _client().post(
        "/api/skill-lab/runs",
        json={
            "workspace_id": "workspace-a",
            "skill_id": skill_id,
            "endpoint_id": "hermes",
            "cases": [
                {
                    "case_id": "positive",
                    "prompt": "one",
                    "expected_patterns": ["DONE"],
                    "should_invoke": True,
                },
                {
                    "case_id": "negative",
                    "prompt": "two",
                    "expected_patterns": ["DONE"],
                    "should_invoke": False,
                },
            ],
        },
    )
    run = _client().get(f"/api/skill-lab/runs/{response.json()['run_id']}").json()
    assert run["kpis"]["success_uplift"] == 1.0
    assert run["kpis"]["routing_precision"] == 1
    assert run["kpis"]["sample_counts"]["total"] == 2

    responses_without_marker = True
    failed_closed = _client().post(
        "/api/skill-lab/runs",
        json={
            "workspace_id": "workspace-a",
            "skill_id": skill_id,
            "endpoint_id": "hermes",
            "cases": [{"prompt": "missing marker", "expected_patterns": ["DONE"]}],
        },
    )
    closed_run = _client().get(f"/api/skill-lab/runs/{failed_closed.json()['run_id']}").json()
    assert closed_run["status"] == "completed"
    assert closed_run["kpis"]["verdict"] == "fail"
    assert closed_run["kpis"]["sample_counts"]["invalid_markers"] == 1
    assert all(attempt["passed"] is False for attempt in closed_run["attempts"])


def test_skill_detail_exposes_revision_and_structured_frontmatter(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    skill_id = _make_skill(
        monkeypatch,
        tmp_path,
        "---\nname: detail-skill\ndescription: Details\nallowed_tools:\n  - Read\n  - Bash\n---\nBody\n",
    )
    response = _client().get(f"/api/skill-lab/skills/{skill_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["skill_id"] == skill_id
    assert data["revision"] == data["content_revision"]
    assert data["frontmatter"]["description"] == "Details"
    assert data["allowed_tools"] == ["Read", "Bash"]
    assert data["content"].endswith("Body\n")


def test_plan_generation_builds_balanced_cases_without_workspace_tools(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    skill_id = _make_skill(
        monkeypatch,
        tmp_path,
        "---\nname: review-helper\ndescription: Reviews patches\n---\nFind regressions.\n",
    )
    observed = {}

    async def fake_chat(endpoint_id, request):
        observed["endpoint_id"] = endpoint_id
        observed["request"] = request
        return {
            "model": "planner-model",
            "response": """```json
            {"cases": [
              {"name":"Regression","prompt":"Review this patch","should_invoke":true,"expected_patterns":["regression|finding"],"forbidden_patterns":[]},
              {"name":"Tests","prompt":"Check this change for missing tests","should_invoke":true,"expected_patterns":["test|coverage"],"forbidden_patterns":[]},
              {"name":"Weather","prompt":"What is the weather?","should_invoke":false,"expected_patterns":["\\\\S"],"forbidden_patterns":[]},
              {"name":"Recipe","prompt":"Suggest a dinner recipe","should_invoke":false,"expected_patterns":["\\\\S"],"forbidden_patterns":[]}
            ]}
            ```""",
        }

    monkeypatch.setattr(agent_routes, "chat_with_agent_endpoint", fake_chat)
    response = _client().post(
        "/api/skill-lab/plans",
        json={
            "workspace_id": "workspace-a",
            "skill_id": skill_id,
            "endpoint_id": "hermes",
            "goal": "Catch behavioral regressions",
            "depth": "quick",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "Catch behavioral regressions"
    assert data["evaluation_calls"] == 12
    assert data["generation_model"] == "planner-model"
    assert [case["should_invoke"] for case in data["cases"]] == [True, True, False, False]
    assert observed["endpoint_id"] == "hermes"
    assert observed["request"].enable_skill_lab_tools is False


def test_skill_file_changes_require_confirmation_and_revision_match(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    skills_root = repo_root / "assistant" / "workspace" / "skills"
    monkeypatch.setattr(agent_routes, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        agent_routes,
        "_toolkit_skill_root_candidates",
        lambda: [("workspace", skills_root)],
    )
    agent_routes._TOOLKIT_CACHE.clear()

    draft = {
        "name": "Prompt Gardener",
        "description": "Tends prompts",
        "content": "# Prompt Gardener\n\nKeep prompts concise.",
        "confirmed": False,
    }
    assert _client().post("/api/skill-lab/skills", json=draft).status_code == 409

    created = _client().post("/api/skill-lab/skills", json={**draft, "confirmed": True})
    assert created.status_code == 201
    skill = created.json()
    assert skill["source"] == "workspace"
    path = skills_root / "prompt-gardener" / "SKILL.md"
    assert path.exists()
    assert 'description: "Tends prompts"' in path.read_text(encoding="utf-8")

    stale = _client().put(
        f"/api/skill-lab/skills/{skill['skill_id']}",
        json={
            "content": "# Updated\n",
            "expected_revision": "stale",
            "confirmed": True,
        },
    )
    assert stale.status_code == 409

    updated = _client().put(
        f"/api/skill-lab/skills/{skill['skill_id']}",
        json={
            "content": "---\nname: Prompt Gardener\n---\n\n# Updated\n",
            "expected_revision": skill["revision"],
            "confirmed": True,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["skill_id"] == skill["skill_id"]
    assert updated.json()["revision"] != skill["revision"]


def test_source_managed_bashgym_skills_cannot_be_self_modified(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "bashgym-operator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: bashgym-operator\ndescription: Critical operator\n---\nBody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent_routes,
        "_toolkit_skill_root_candidates",
        lambda: [("workspace", skills_root)],
    )
    skill = next(
        item for item in skill_lab_routes._skill_inventory() if item.name == "bashgym-operator"
    )

    response = _client().put(
        f"/api/skill-lab/skills/{skill.skill_id}",
        json={
            "content": "---\nname: bashgym-operator\n---\nRewritten\n",
            "expected_revision": skill.revision,
            "confirmed": True,
        },
    )

    assert response.status_code == 403
    assert "source-managed critical BashGym skill" in response.json()["detail"]
    assert "Rewritten" not in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_inventory_uses_a_fair_cap_and_collapses_exact_duplicates(monkeypatch, tmp_path):
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    for root, names in ((root_a, ("shared", "alpha")), (root_b, ("shared", "beta"))):
        for name in names:
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "SKILL.md").write_text(
                f"---\nname: {name}\nallowed_tools: [Read]\n---\nSame body\n",
                encoding="utf-8",
            )
    monkeypatch.setattr(
        agent_routes,
        "_toolkit_skill_root_candidates",
        lambda: [("first", root_a), ("second", root_b)],
    )

    roots, skills, warnings = agent_routes._scan_skill_roots(max_skills=2)
    local = [skill for skill in skills if skill.source in {"first", "second"}]
    assert [root.skill_count for root in roots] == [2, 2]
    assert len(local) == 3
    shared = next(skill for skill in local if skill.name == "shared")
    assert shared.skill_id == agent_routes._stable_skill_id(
        "shared", "first", root_a / "shared" / "SKILL.md"
    )
    assert shared.allowed_tools == ["Read"]
    assert str(root_b / "shared" / "SKILL.md") in shared.shadowed_paths
    assert not warnings


def test_inventory_revision_changes_without_changing_skill_id(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    directory = root / "stable"
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text("---\nname: stable\n---\nfirst\n", encoding="utf-8")
    monkeypatch.setattr(agent_routes, "_toolkit_skill_root_candidates", lambda: [("test", root)])
    first = next(skill for skill in agent_routes._scan_skill_roots()[1] if skill.source == "test")

    path.write_text("---\nname: stable\n---\nsecond\n", encoding="utf-8")
    second = next(skill for skill in agent_routes._scan_skill_roots()[1] if skill.source == "test")
    assert second.skill_id == first.skill_id
    assert second.revision != first.revision


def test_inventory_distinguishes_same_name_at_different_paths(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    for dirname, body in (("first", "one"), ("second", "two")):
        directory = root / dirname
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: shared-name\n---\n{body}\n", encoding="utf-8"
        )
    monkeypatch.setattr(agent_routes, "_toolkit_skill_root_candidates", lambda: [("test", root)])

    skills = [skill for skill in agent_routes._scan_skill_roots()[1] if skill.source == "test"]

    assert len(skills) == 2
    assert len({skill.skill_id for skill in skills}) == 2


def test_inventory_revision_includes_packaged_scripts(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    directory = root / "scripted"
    scripts = directory / "scripts"
    scripts.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: scripted\n---\nBody\n", encoding="utf-8")
    script = scripts / "run.py"
    script.write_text("print('one')\n", encoding="utf-8")
    monkeypatch.setattr(agent_routes, "_toolkit_skill_root_candidates", lambda: [("test", root)])
    first = next(skill for skill in agent_routes._scan_skill_roots()[1] if skill.source == "test")

    script.write_text("print('two')\n", encoding="utf-8")
    second = next(skill for skill in agent_routes._scan_skill_roots()[1] if skill.source == "test")

    assert second.skill_id == first.skill_id
    assert second.revision != first.revision


def test_inventory_prunes_host_mirrors_and_classifies_catalog_entries(monkeypatch, tmp_path):
    root = tmp_path / "skills"

    direct = root / "review"
    direct.mkdir(parents=True)
    (direct / "SKILL.md").write_text(
        "---\nname: review\ndescription: Canonical review\n---\nDirect instructions.\n",
        encoding="utf-8",
    )

    alternate = root / "bundle" / "review"
    alternate.mkdir(parents=True)
    (alternate / "SKILL.md").write_text(
        "---\nname: review\ndescription: Older review\n---\nBundled instructions.\n",
        encoding="utf-8",
    )

    mirrored = root / "bundle" / ".hermes" / "skills" / "review"
    mirrored.mkdir(parents=True)
    (mirrored / "SKILL.md").write_text(
        "---\nname: review\ndescription: Host mirror\n---\nMirrored instructions.\n",
        encoding="utf-8",
    )

    archived = root / "_archive" / "retired"
    archived.mkdir(parents=True)
    (archived / "SKILL.md").write_text(
        "---\nname: retired\ndescription: Retired\n---\nOld instructions.\n",
        encoding="utf-8",
    )

    deprecated = root / "deprecated-skill"
    deprecated.mkdir(parents=True)
    (deprecated / "SKILL.md").write_text(
        "---\nname: deprecated-skill\ndescription: Deprecated\ndeprecated: true\n---\nOld.\n",
        encoding="utf-8",
    )

    invalid = root / "invalid-skill"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text("# Invalid skill\n", encoding="utf-8")

    monkeypatch.setattr(
        agent_routes,
        "_toolkit_skill_root_candidates",
        lambda: [("test", root)],
    )

    roots, skills, warnings = agent_routes._scan_skill_roots()

    assert warnings == []
    assert roots[0].skill_count == 4
    assert all(".hermes" not in str(skill.path) for skill in skills if skill.path)
    assert all("_archive" not in str(skill.path) for skill in skills if skill.path)
    reviews = [skill for skill in skills if skill.name == "review"]
    canonical = next(skill for skill in reviews if skill.catalog_status == "active")
    alternate_skill = next(skill for skill in reviews if skill.catalog_status == "alternate")
    assert canonical.path == str(direct / "SKILL.md")
    assert alternate_skill.canonical_skill_id == canonical.skill_id
    assert alternate_skill.path in canonical.shadowed_paths
    assert (
        next(skill for skill in skills if skill.name == "deprecated-skill").catalog_status
        == "deprecated"
    )
    invalid_skill = next(skill for skill in skills if skill.name == "Invalid skill")
    assert invalid_skill.catalog_status == "invalid"
    assert invalid_skill.quality_issues == ["missing_frontmatter", "missing_description"]
