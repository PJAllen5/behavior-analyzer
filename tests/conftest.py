"""Shared test fixtures.

Test movies are made by the tools/make_test_*.py generators and cached in tests/.data/
(regenerated automatically when a generator changes). Each mode is analyzed once per
test run, and the results are shared by the tests that need them.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
CACHE = ROOT / "tests" / ".data"
sys.path.insert(0, str(TOOLS))  # so tests can import the answer-key checkers

# name -> (generator script, file the generator writes, given the base path)
GENERATORS = {
    "flock": ("make_test_video.py", lambda base: base.with_suffix(".mp4"), lambda base: base.with_suffix(".mp4")),
    "cells": ("make_test_cells.py", lambda base: base, lambda base: base.with_suffix(".tif")),
    "neurons": ("make_test_neurons.py", lambda base: base, lambda base: base.with_suffix(".tif")),
    "neurites": ("make_test_neurites.py", lambda base: base, lambda base: base.with_suffix(".tif")),
}


def make_test_data(name: str) -> tuple[Path, Path]:
    """Return (movie file, answer-key base path), generating them if needed."""
    script, arg, movie = GENERATORS[name]
    base = CACHE / name / f"test_{name}"
    stamp = base.parent / "generator.sha1"
    digest = hashlib.sha1((TOOLS / script).read_bytes()).hexdigest()
    if not (movie(base).exists() and stamp.exists() and stamp.read_text() == digest):
        shutil.rmtree(base.parent, ignore_errors=True)
        base.parent.mkdir(parents=True)
        subprocess.run([sys.executable, str(TOOLS / script), str(arg(base))], check=True,
                       capture_output=True, cwd=ROOT)
        stamp.write_text(digest)
    return movie(base), base


# mode name -> (test data, settings mode, neural analysis)
ANALYSES = {
    "flock": ("flock", "animals", None),
    "cells": ("cells", "cells", None),
    "neurons": ("neurons", "neurons", "activity"),
    "neurites": ("neurites", "neurons", "neurites"),
}


@pytest.fixture(scope="session")
def analyzed(tmp_path_factory):
    """analyzed("cells") -> dict(movie, truth, out, summary). Each runs once per session."""
    from behavetrack.pipeline import run
    from behavetrack.settings import defaults

    done: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in done:
            data, mode, analysis = ANALYSES[name]
            movie, truth = make_test_data(data)
            out = tmp_path_factory.mktemp(f"results_{name}") / f"test_{name}"
            summary = run(movie, out, defaults(mode, analysis))
            done[name] = {"movie": movie, "truth": truth, "out": out, "summary": summary}
        return done[name]

    return get
