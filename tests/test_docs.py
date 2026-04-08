from __future__ import annotations

from pathlib import Path


def test_readme_contains_core_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "openenv validate" in readme
    assert "docker build" in readme
    assert "python3 inference.py" in readme
