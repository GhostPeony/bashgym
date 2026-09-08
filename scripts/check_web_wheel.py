"""Verify a release wheel locates its bundled SPA outside the source checkout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check(wheel: Path) -> dict[str, int]:
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as temporary:
        names = archive.namelist()
        assert "bashgym/web_assets/index.html" in names, "Release wheel is missing the browser"
        assets = [name for name in names if name.startswith("bashgym/web_assets/")]
        assert any(name.endswith(".js") for name in assets), "Missing browser JavaScript"
        html = archive.read("bashgym/web_assets/index.html").decode("utf-8")
        assert not re.search(r"https://fonts\.(googleapis|gstatic)\.com", html)
        entry_scripts = re.findall(r'<script\b[^>]*src="([^"]+\.js)"', html)
        assert entry_scripts, "Missing browser entry script"
        entry_bytes = sum(
            len(archive.read("bashgym/web_assets/" + source.lstrip("/")))
            for source in entry_scripts
        )
        for font in ("Fraunces.ttf", "SourceSans3.ttf", "IBMPlexMono-Regular.ttf"):
            assert any(name.endswith("/" + font) for name in assets), f"Missing font: {font}"
        target = Path(temporary)
        for name in names:
            assert (target / name).resolve().is_relative_to(target.resolve()), "Unsafe wheel path"
        archive.extractall(target)
        probe = (
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from bashgym.web_assets import frontend_directory; "
            "p = frontend_directory(); "
            "assert p is not None and (p / 'index.html').is_file(); "
            "assert p.resolve().is_relative_to(Path(sys.argv[1]).resolve()); "
            "assert any(p.rglob('*.js'))"
        )
        subprocess.run(
            [sys.executable, "-I", "-c", probe, temporary],
            cwd=temporary,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {
            "wheel_bytes": wheel.stat().st_size,
            "web_asset_files": len(assets),
            "entry_javascript_bytes": entry_bytes,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    print(json.dumps(check(parser.parse_args().wheel)))
