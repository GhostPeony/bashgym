"""Locate browser assets in an installed release or a development checkout."""

from pathlib import Path


def frontend_directory() -> Path:
    packaged = Path(__file__).parent / "web_assets"
    if (packaged / "index.html").is_file():
        return packaged
    return Path(__file__).parent.parent / "frontend" / "dist"
