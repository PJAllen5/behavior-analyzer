"""The command line and the app must use the same defaults (from behavetrack/settings.py)."""
from __future__ import annotations

import argparse
from dataclasses import asdict

import pytest
from streamlit.testing.v1 import AppTest

from behavetrack import cli
from behavetrack.settings import Settings, defaults, with_changes

APP = str(__import__("pathlib").Path(cli.__file__).with_name("app.py"))

CASES = [
    # (CLI arguments, app mode pickers, settings mode, neural analysis)
    ([], {"subject": "Animals or people"}, "animals", None),
    (["--mode", "cells"], {"subject": "Cells (microscope)"}, "cells", None),
    (["--mode", "neurons"], {"subject": "Neurons (microscope)", "neural_analysis": "Firing activity"},
     "neurons", "activity"),
    (["--mode", "neurons", "--analysis", "neurites"],
     {"subject": "Neurons (microscope)", "neural_analysis": "Neurite growth"}, "neurons", "neurites"),
]


def cli_settings(argv: list[str]) -> Settings:
    p = argparse.ArgumentParser()
    cli._add_common(p)
    return cli._settings(p.parse_args(argv))


@pytest.mark.parametrize("argv,_picks,mode,analysis", CASES)
def test_cli_defaults_match_settings(argv, _picks, mode, analysis):
    assert cli_settings(argv) == defaults(mode, analysis)


@pytest.mark.parametrize("_argv,picks,mode,analysis", CASES)
def test_app_defaults_match_settings(_argv, picks, mode, analysis, tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVETRACK_DATA", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    for key, value in picks.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    got, want = asdict(at.session_state["analysis_settings"]), asdict(defaults(mode, analysis))
    assert {k: v for k, v in got.items() if v != want[k]} == {}


@pytest.mark.parametrize("argv,field,value", [
    (["--mode", "cells", "--contact-bl", "3"], "contact_bl", 3.0),
    (["--mode", "cells", "--no-divisions"], "detect_divisions", False),
    (["--mode", "neurons", "--event-sd", "5"], "event_threshold_sd", 5.0),
    (["--mode", "neurons", "--analysis", "neurites", "--neurite-width", "4.5"], "neurite_width_px", 4.5),
    (["--sensitivity", "40", "--no-video"], "sensitivity", 40.0),
    (["--no-video"], "annotate", False),
    (["--classes", "bird, dog"], "classes", ["bird", "dog"]),
])
def test_cli_options_change_only_their_setting(argv, field, value):
    s = cli_settings(argv)
    assert getattr(s, field) == value
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "animals"
    analysis = argv[argv.index("--analysis") + 1] if "--analysis" in argv else None
    base = asdict(defaults(mode, analysis))
    changed = {k for k, v in asdict(s).items() if v != base[k]}
    assert changed <= {field, "annotate", "sensitivity"}


def test_every_cli_option_maps_to_a_real_setting():
    names = set(asdict(Settings()))
    assert set(cli._OPTION_FIELDS.values()) <= names
    assert {f for f, _ in cli._SWITCHES.values()} <= names


def test_with_changes_ignores_none_and_rejects_typos():
    base = defaults("cells")
    assert with_changes(base, pixel_size_um=None) == base
    with pytest.raises(TypeError):
        with_changes(base, pixel_sise_um=1.0)
    with pytest.raises(ValueError):
        defaults("fish")
