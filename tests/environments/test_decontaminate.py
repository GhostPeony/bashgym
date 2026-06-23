"""Tests for benchmark decontamination over EnvironmentSpec."""

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.decontaminate import environment_text, filter_contaminated_environments


def test_environment_text_contains_instruction_and_files():
    env = EnvironmentSpec(
        id="env_a",
        instruction="Repair the exact benchmark task.",
        files={"README.md": "important setup notes"},
    )

    text = environment_text(env)

    assert "Repair the exact benchmark task." in text
    assert "README.md" in text
    assert "important setup notes" in text


def test_filter_contaminated_environments_drops_overlap():
    benchmark = "install the package and run the tests before reporting success"
    clean = EnvironmentSpec(id="clean", instruction="Summarize a local log file.")
    leaked = EnvironmentSpec(id="leaked", instruction=benchmark)

    kept, report = filter_contaminated_environments([clean, leaked], [benchmark])

    assert [env.id for env in kept] == ["clean"]
    assert report.kept == 1
    assert report.dropped == 1
