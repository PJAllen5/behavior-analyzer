"""Exporting results for complexity-engine, and running it."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from behavetrack.export import ENGINE_RUNS, engine_available, export, parse_input, run_engine

needs_engine = pytest.mark.skipif(not engine_available(), reason="complexity-engine not installed")


def _fake_results(root: Path, name: str, mode: str = "animals", n: int = 60) -> Path:
    """A tiny results folder with the files the export reads."""
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"mode": mode}))
    pd.DataFrame({"frame": range(n), "time_s": [f / 10 for f in range(n)],
                  "n_visible": 5, "polarization": [0.5] * n, "rotation": [0.1] * n,
                  "mean_speed_bl_s": [1.0] * n, "nn_dist_bl": [2.0] * n,
                  "spread_bl": [3.0] * n, "behavior": "disordered"}).to_csv(d / "frames.csv", index=False)
    pd.DataFrame({"id_a": [1, 2, 1], "id_b": [2, 1, 3], "duration_s": [1.0, 0.5, 2.0]}).to_csv(
        d / "interactions.csv", index=False)
    return d


def test_parse_input_reads_optional_condition(tmp_path):
    d = _fake_results(tmp_path, "video1")
    assert parse_input(str(d)) == (d, "video1")
    assert parse_input(f"{d}:control") == (d, "control")


def test_export_shapes_and_conditions(tmp_path):
    a = _fake_results(tmp_path, "a")
    b = _fake_results(tmp_path, "b")
    r = export([f"{a}:treated", f"{b}:treated"], tmp_path / "out")
    ts = pd.read_csv(r.files["collective_timeseries.csv"])
    assert list(ts.columns[:3]) == ["time", "session", "condition"]
    assert set(ts.session) == {"a", "b"} and set(ts.condition) == {"treated"}  # grouped for comparison
    net = pd.read_csv(r.files["contact_network.csv"])
    # every video is its own network; repeated contacts in either order collapse into one weighted tie
    assert set(net.condition) == {"treated / a", "treated / b"}
    one = net[net.condition == "treated / a"].set_index(["source", "target"])
    assert one.loc[(1, 2), "weight"] == pytest.approx(1.5) and one.loc[(1, 2), "contacts"] == 2


def test_non_results_folders_are_skipped(tmp_path):
    (tmp_path / "empty").mkdir()
    r = export([str(tmp_path / "empty")], tmp_path / "out")
    assert not r.files and "no summary.json" in r.skipped[0]


def test_command_line_export(tmp_path):
    d = _fake_results(tmp_path, "clip")
    out = tmp_path / "cx"
    run = subprocess.run([sys.executable, "-m", "behavetrack", "export-complexity", f"{d}:wild", "-o", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (out / "collective_timeseries.csv").exists()
    assert "--domain collective_behavior" in run.stdout  # tells you how to run the engine


@pytest.mark.slow
def test_export_every_mode(analyzed, tmp_path):
    names = ["flock", "cells", "neurons", "neurites"]
    results = {n: analyzed(n)["out"] for n in names}
    r = export([f"{results['flock']}:fish", f"{results['cells']}:cells", f"{results['neurons']}:control",
                str(results["neurites"])], tmp_path / "out", min_correlation=0.5)
    assert set(r.files) == set(ENGINE_RUNS)
    assert any("neurite growth" in m for m in r.skipped)

    ts = pd.read_csv(r.files["collective_timeseries.csv"])
    assert set(ts.condition) == {"fish", "cells"}
    neural = pd.read_csv(r.files["neural_timeseries.csv"])
    n_neurons = len(pd.read_csv(results["neurons"] / "neurons.csv"))
    assert neural.neuron_id.nunique() == n_neurons
    assert {"dff", "fraction_firing"} <= set(neural.columns)
    func = pd.read_csv(r.files["functional_network.csv"])
    assert (func.weight >= 0.5).all()


@pytest.mark.slow
@needs_engine
def test_run_engine_with_behavior_profiles(analyzed, tmp_path):
    flock, neurons = analyzed("flock")["out"], analyzed("neurons")["out"]
    out = tmp_path / "out"
    r = run_engine(export([str(flock), str(neurons)], out), out)
    assert set(r.reports) == set(ENGINE_RUNS)
    for file, folder in r.reports.items():
        report = (folder / "report.md").read_text(encoding="utf-8")
        assert ENGINE_RUNS[file][1] in report  # the matching profile was used
        assert (folder / "report.html").exists()
    summary = pd.read_csv(r.reports["collective_timeseries.csv"] / "complexity_summary.csv")
    assert summary["dfa_exponent"].notna().any()


@pytest.mark.slow
@needs_engine
def test_app_runs_complexity_analysis(analyzed, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    res = analyzed("flock")["out"]
    data = tmp_path / "app_data"
    shutil.copytree(res, data / "results" / res.name)
    monkeypatch.setenv("BEHAVETRACK_DATA", str(data))
    app = str(Path(__file__).resolve().parents[1] / "behavetrack" / "app.py")
    at = AppTest.from_file(app, default_timeout=180)
    at.session_state["selected"] = res.name
    at.run()
    assert not at.exception
    run_button = next(b for b in at.button if b.label == "Run complexity analysis")
    run_button.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert (data / "results" / res.name / "complexity" / "reports" / "contact_network" / "report.html").exists()
    assert any("collective_behavior" in e.label for e in at.expander)
