import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "assistant"
    / "workspace"
    / "skills"
    / "bashgym-operator"
    / "scripts"
    / "gbrain_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("bashgym_gbrain_bridge", SCRIPT)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "bridge.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": bridge.PROFILE_SCHEMA,
                "ssh_target": "training-host",
                "activity_root": "~/brain-sources/bashgym-activity",
                "gbrain_bin": "~/gbrain/bin/gbrain",
                "source_id": "bashgym-activity",
            }
        ),
        encoding="utf-8",
    )
    return path


def _receipt(tmp_path: Path) -> Path:
    path = tmp_path / "handoff.md"
    path.write_text(
        '---\nschema_version: "bashgym.activity.v1"\n'
        'privacy: "full-tier-only"\n---\n\n# Safe handoff\n',
        encoding="utf-8",
    )
    return path


def test_publish_is_dry_run_by_default(tmp_path, monkeypatch):
    profile = bridge._profile(_profile(tmp_path))
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("SSH must not run during preview")

    monkeypatch.setattr(bridge, "_remote_python", fail_if_called)
    result = bridge._publish(
        profile,
        document=_receipt(tmp_path),
        relative="session-handoffs/workspace/project/session.md",
        execute=False,
        sync=True,
    )

    assert result["executed"] is False
    assert result["synced"] is False
    assert result["indexed"] is False
    assert called is False


def test_bridge_rejects_paths_and_unrendered_documents(tmp_path):
    profile = bridge._profile(_profile(tmp_path))
    document = tmp_path / "plain.md"
    document.write_text("raw transcript", encoding="utf-8")

    try:
        bridge._publish(
            profile,
            document=document,
            relative="../escape.md",
            execute=False,
            sync=False,
        )
    except ValueError as exc:
        assert "rendered BashGym activity receipt" in str(exc)
    else:
        raise AssertionError("unrendered documents must be rejected")

    assert bridge._relative_document("session-handoffs/workspace/project/session.md")
    for unsafe in ("../escape.md", "/absolute.md", "handoffs/raw.txt"):
        try:
            bridge._relative_document(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {unsafe}")


def test_bridge_full_rescans_the_bounded_source_after_atomic_publish():
    assert "'--full'" in bridge.REMOTE_GBRAIN
    assert "'--yes'" in bridge.REMOTE_GBRAIN
    assert "relative not in output" in bridge.REMOTE_VERIFY
    assert (
        bridge.PurePosixPath("handoffs/session.md").with_suffix("").as_posix() == "handoffs/session"
    )
