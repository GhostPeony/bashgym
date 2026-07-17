import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[2] / "scripts" / "rebuild_training_data.py"
_SPEC = importlib.util.spec_from_file_location("rebuild_training_data", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
rebuild_training_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rebuild_training_data)


def test_project_name_from_session_path_strips_windows_user_identity():
    source = Path("C--Users-Developer-projects-example-app") / "session.jsonl"

    assert rebuild_training_data.project_name_from_session_path(source) == "example-app"


def test_project_name_from_session_path_strips_posix_user_identity():
    source = Path("-home-developer-projects-example-app") / "session.jsonl"

    assert rebuild_training_data.project_name_from_session_path(source) == "example-app"


def test_project_name_from_session_path_labels_encoded_home_without_identity():
    source = Path("C--Users-Developer") / "session.jsonl"

    assert rebuild_training_data.project_name_from_session_path(source) == "home"


def test_project_name_from_session_path_preserves_unrecognized_project_directory():
    source = Path("shared-project") / "session.jsonl"

    assert rebuild_training_data.project_name_from_session_path(source) == "shared-project"
