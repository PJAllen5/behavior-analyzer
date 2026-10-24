"""The installed commands work end to end."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from behavetrack import cli


def _behavetrack(*args, cwd=None):
    """Run the installed `behavetrack` command (falls back to python -m behavetrack)."""
    exe = shutil.which("behavetrack", path=str(Path(sys.executable).parent))
    cmd = [exe, *args] if exe else [sys.executable, "-m", "behavetrack", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def test_help_lists_all_commands_and_modes():
    out = _behavetrack("--help")
    assert out.returncode == 0
    for word in ("app", "analyze", "filter"):
        assert word in out.stdout
    out = _behavetrack("analyze", "--help")
    assert out.returncode == 0
    assert "animals,cells,neurons" in out.stdout.replace(" ", "")


def test_app_command_starts_streamlit_with_the_packaged_app(monkeypatch):
    seen = {}

    def fake_main():
        seen["argv"] = list(sys.argv)
        return 0

    from streamlit.web import cli as stcli
    monkeypatch.setattr(stcli, "main", fake_main)
    with pytest.raises(SystemExit):
        cli.main(["app", "--no-browser", "--port", "8555"])
    argv = seen["argv"]
    assert argv[:2] == ["streamlit", "run"]
    assert Path(argv[2]) == Path(cli.__file__).with_name("app.py")
    assert argv[argv.index("--server.port") + 1] == "8555"
    assert argv[argv.index("--server.headless") + 1] == "true"


@pytest.mark.slow
def test_analyze_and_filter_from_the_command_line(tmp_path):
    from conftest import make_test_data

    movie, _ = make_test_data("flock")
    out = tmp_path / "flock_results"
    r = _behavetrack("analyze", str(movie), "-o", str(out), "--no-video", "--max-seconds", "12")
    assert r.returncode == 0, r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["analyzed_seconds"] == pytest.approx(12, abs=0.1)
    assert not (out / "annotated.mp4").exists()

    search = tmp_path / "search"
    shutil.copy(movie, tmp_path / movie.name)
    r = _behavetrack("filter", str(tmp_path / movie.name), "--behavior", "collective_motion",
                     "-o", str(search), "--clips", "--no-video")
    assert r.returncode == 0, r.stderr
    assert "collective_motion" in (search / "matches_collective_motion.csv").read_text()
    assert list((search / "clips").glob("*.mp4"))
