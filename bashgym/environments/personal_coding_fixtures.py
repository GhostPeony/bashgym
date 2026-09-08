"""Authored, versioned coding fixtures; no user traces or external benchmark data."""

VERIFIER_SOURCE = """import importlib.util
import json
import pathlib
import sys
import unittest

root = pathlib.Path(__file__).resolve().parent
support_spec = importlib.util.spec_from_file_location("coding_test_support", root / "coding_test_support.py")
support = importlib.util.module_from_spec(support_spec)
sys.modules["coding_test_support"] = support
support_spec.loader.exec_module(support)
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
report = {
    "schema_version": "bashgym.coding_tests.v1",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}
print(json.dumps(report, sort_keys=True))
sys.exit(0 if result.testsRun > 0 and result.wasSuccessful() and not result.skipped else 1)
"""

TEST_SUPPORT_SOURCE = '''import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
CHILD_SOURCE = """import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arguments = json.loads(sys.argv[3])
result = getattr(module, sys.argv[2])(*arguments)
print(json.dumps({'result': result, 'arguments': arguments}))
"""

def call_solution(function, arguments):
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-I", "-c", CHILD_SOURCE, str(ROOT / "solution.py"), function, json.dumps(arguments)],
        capture_output=True, text=True, timeout=3,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise AssertionError("candidate execution failed: " + completed.stderr[:1000])
    response = json.loads(completed.stdout)
    if response.get("arguments") != arguments:
        raise AssertionError("candidate mutated its input")
    return response["result"]
'''

TASKS = (
    {
        "id": "personal-coding-v1-train-slug",
        "split": "train",
        "task_family": "repository_repair",
        "instruction": "Repair slug(text) in solution.py: lowercase words, collapse all whitespace to one hyphen, and omit leading/trailing separators. Preserve other characters. Do not edit the tests or verifier.",
        "files": {"solution.py": "def slug(text):\n    return text.lower().replace(' ', '-')\n"},
        "tests": """import unittest
from coding_test_support import call_solution
def slug(text): return call_solution("slug", [text])
class Tests(unittest.TestCase):
    def test_edges(self): self.assertEqual(slug("  HELLO   World  "), "hello-world")
    def test_whitespace(self): self.assertEqual(slug("one\\ttwo\\nthree"), "one-two-three")
    def test_empty(self): self.assertEqual(slug("   "), "")
    def test_punctuation(self): self.assertEqual(slug("A_B!"), "a_b!")
""",
        "test_count": 4,
    },
    {
        "id": "personal-coding-v1-train-config",
        "split": "train",
        "task_family": "tool_recovery",
        "instruction": "The service startup command python main.py failed because settings.json is missing. Supply settings.json with enabled=true and prefix=ready so the startup prints ready:ok. Keep main.py and the protected tests unchanged.",
        "initial_command": "python main.py",
        "files": {
            "main.py": "import json\nfrom pathlib import Path\nsettings = json.loads(Path('settings.json').read_text())\nif not settings['enabled']:\n    raise RuntimeError('service disabled')\nprint(settings['prefix'] + ':ok')\n"
        },
        "protected_paths": ["main.py"],
        "tests": """import json
import pathlib
import subprocess
import sys
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_valid_json(self): self.assertIsInstance(json.loads((ROOT / "settings.json").read_text()), dict)
    def test_enabled(self): self.assertIs(json.loads((ROOT / "settings.json").read_text())["enabled"], True)
    def test_startup(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "main.py"], cwd=ROOT, capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "ready:ok"))
""",
        "test_count": 3,
    },
    {
        "id": "personal-coding-v1-dev-merge",
        "split": "dev",
        "task_family": "repository_repair",
        "instruction": "Repair merge_counts(left, right) in solution.py. Return a new dictionary that adds counts for overlapping keys, retains all other keys, and does not mutate either input. Negative and zero counts are valid. Do not edit tests or the verifier.",
        "files": {"solution.py": "def merge_counts(left, right):\n    return {**left, **right}\n"},
        "tests": """import unittest
from coding_test_support import call_solution
def merge_counts(left, right): return call_solution("merge_counts", [left, right])
class Tests(unittest.TestCase):
    def test_overlap(self): self.assertEqual(merge_counts({"a": 2}, {"a": 3}), {"a": 5})
    def test_disjoint(self): self.assertEqual(merge_counts({"a": 1}, {"b": 2}), {"a": 1, "b": 2})
    def test_negative(self): self.assertEqual(merge_counts({"x": -2}, {"x": 2}), {"x": 0})
    def test_immutable(self):
        left, right = {"x": 1}, {"x": 4}
        result = merge_counts(left, right)
        self.assertEqual((left, right, result), ({"x": 1}, {"x": 4}, {"x": 5}))
""",
        "test_count": 4,
    },
    {
        "id": "personal-coding-v1-dev-csv",
        "split": "dev",
        "task_family": "tool_recovery",
        "instruction": "The command python main.py scores.csv failed on a CSV header. Repair main.py to sum the integer score column of a UTF-8 CSV file named by its first argument, printing one integer. A header-only CSV sums to zero; negative scores are valid. Preserve the fixture and tests.",
        "initial_command": "python main.py scores.csv",
        "files": {
            "main.py": "import sys\nfrom pathlib import Path\nprint(sum(int(line) for line in Path(sys.argv[1]).read_text().splitlines()))\n",
            "scores.csv": "name,score\nAda,4\nLin,7\n",
        },
        "protected_paths": ["scores.csv"],
        "tests": """import pathlib
import subprocess
import sys
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def check(self, text, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "input.csv"
            path.write_text(text, encoding="utf-8")
            result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "main.py"), str(path)], capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual((result.returncode, result.stdout.strip()), (0, str(expected)))
    def test_rows(self): self.check("name,score\\nAda,4\\nLin,7\\n", 11)
    def test_empty(self): self.check("name,score\\n", 0)
    def test_signed(self): self.check("name,score\\nA,-3\\nB,2\\n", -1)
""",
        "test_count": 3,
    },
    {
        "id": "personal-coding-v1-confirmation-unique",
        "split": "confirmation",
        "task_family": "repository_repair",
        "instruction": "Repair unique(items) in solution.py to remove duplicate hashable values while preserving their first occurrence order. Return a list without modifying the input. Do not edit tests or the verifier.",
        "files": {"solution.py": "def unique(items):\n    return sorted(set(items))\n"},
        "tests": """import unittest
from coding_test_support import call_solution
def unique(items): return call_solution("unique", [items])
class Tests(unittest.TestCase):
    def test_order(self): self.assertEqual(unique([3, 1, 3, 2]), [3, 1, 2])
    def test_strings(self): self.assertEqual(unique(["z", "a", "z"]), ["z", "a"])
    def test_empty(self): self.assertEqual(unique([]), [])
    def test_immutable(self):
        values = [2, 1, 2]
        self.assertEqual(unique(values), [2, 1])
        self.assertEqual(values, [2, 1, 2])
""",
        "test_count": 4,
    },
    {
        "id": "personal-coding-v1-confirmation-unicode",
        "split": "confirmation",
        "task_family": "tool_recovery",
        "instruction": "The command python main.py failed because it looks for text.txt in the current directory. Repair main.py to read UTF-8 text from an optional path argument, or assets/text.txt relative to main.py by default, then print its uppercase form. It must work from another working directory. Preserve the asset and tests.",
        "initial_command": "python main.py",
        "files": {
            "main.py": "from pathlib import Path\nprint(Path('text.txt').read_text(encoding='ascii').upper())\n",
            "assets/text.txt": "caf\u00e9\nna\u00efve",
        },
        "protected_paths": ["assets/text.txt"],
        "tests": """import pathlib
import subprocess
import sys
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def check(self, text=None):
        with tempfile.TemporaryDirectory() as directory:
            args = [sys.executable, "-X", "utf8", str(ROOT / "main.py")]
            if text is not None:
                path = pathlib.Path(directory) / "custom.txt"
                path.write_text(text, encoding="utf-8")
                args.append(str(path))
            result = subprocess.run(args, cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual(result.returncode, 0)
            return result.stdout.strip()
    def test_default(self): self.assertEqual(self.check(), "CAF\u00c9\\nNA\u00cfVE")
    def test_argument(self): self.assertEqual(self.check("hello"), "HELLO")
    def test_unicode(self): self.assertEqual(self.check("\u00e9lan"), "\u00c9LAN")
""",
        "test_count": 3,
    },
)
