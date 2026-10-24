"""The browser app's pages load without errors for results from every mode."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "behavetrack" / "app.py")

# result -> a metric label that only that mode's results page shows
PAGES = {
    "flock": "Typical number in view",
    "cells": "Divisions",
    "neurons": "Network bursts",
    "neurites": "Growth rate (µm/h)",
}


def test_app_starts_with_no_results(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVETRACK_DATA", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert (tmp_path / "uploads").is_dir() and (tmp_path / "results").is_dir()


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PAGES))
def test_results_page(name, analyzed, tmp_path, monkeypatch):
    r = analyzed(name)
    data = tmp_path / "app_data"
    shutil.copytree(r["out"], data / "results" / r["out"].name)
    monkeypatch.setenv("BEHAVETRACK_DATA", str(data))
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["selected"] = r["out"].name
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert PAGES[name] in labels, labels
