"""Include the prebuilt browser in release wheels without requiring Node at install."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithWeb(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "frontend" / "dist"
        build_root = Path(self.build_lib).resolve()
        target = build_root / "bashgym" / "web_assets"
        if target.is_symlink() or not target.resolve().is_relative_to(build_root):
            raise ValueError("Browser build destination escapes the build directory")
        if target.exists():
            shutil.rmtree(target)
        if (source / "index.html").is_file():
            shutil.copytree(source, target)
        else:
            self.announce(
                "Building headless wheel: run the frontend web build for browser assets", level=2
            )


setup(cmdclass={"build_py": BuildWithWeb})
