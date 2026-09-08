"""Standalone verification helper embedded into generated training scripts."""


def run_verification(code, test_code, *, detailed=False):
    """Return JUnit outcomes, or the compatible passed/total tuple by default.

    Detailed results distinguish test failures from collection errors, empty
    suites, timeouts, interrupted runners, and unusable reports.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import xml.etree.ElementTree as ET

    def finish(status, counts=None):
        outcome = {"status": status, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        if counts:
            outcome.update(counts)
        outcome["total"] = sum(outcome[key] for key in ("passed", "failed", "errors", "skipped"))
        if detailed:
            return outcome
        passed = outcome["passed"] if status == "completed" else 0
        return passed, max(outcome["total"], 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_solution.py")
        report_path = os.path.join(tmpdir, "results.xml")
        with open(solution_path, "w", encoding="utf-8") as stream:
            stream.write(code)
        with open(test_path, "w", encoding="utf-8") as stream:
            stream.write(test_code)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    test_path,
                    "--tb=no",
                    "-q",
                    "--junitxml",
                    report_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode not in (0, 1, 2, 5):
                return finish("runner_error")
            cases = ET.parse(report_path).getroot().findall(".//testcase")
            if not cases:
                return finish("empty_suite" if result.returncode in (0, 5) else "runner_error")
            # A teardown error can produce another report for the same case.
            outcomes = {}
            for case in cases:
                key = (case.get("classname"), case.get("name"))
                category = "passed"
                for tag, candidate in (
                    ("skipped", "skipped"),
                    ("failure", "failed"),
                    ("error", "errors"),
                ):
                    if case.find(tag) is not None:
                        category = candidate
                priority = {"passed": 0, "skipped": 1, "failed": 2, "errors": 3}
                previous = outcomes.get(key, "passed")
                outcomes[key] = max((previous, category), key=priority.get)
            counts = {
                key: list(outcomes.values()).count(key)
                for key in ("passed", "failed", "errors", "skipped")
            }
            if result.returncode == 2:
                return finish("collection_error" if counts["errors"] else "interrupted", counts)
            if result.returncode not in (0, 1) or (
                result.returncode == 1 and not (counts["failed"] or counts["errors"])
            ):
                return finish("runner_error", counts)
            return finish("completed", counts)
        except subprocess.TimeoutExpired:
            return finish("timeout")
        except (OSError, ET.ParseError):
            return finish("invalid_report")
        except subprocess.SubprocessError:
            return finish("runner_error")
