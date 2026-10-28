"""The Cellpose neuron finder option, and benchmark scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavetrack import activity, cli
from behavetrack.scoring import score_neurofinder

CELLPOSE_MODEL = Path.home() / ".cellpose" / "models" / "cpsam_v2"
# Only run Cellpose where its model is already downloaded, so tests never start a 1.2 GB download.
needs_cellpose = pytest.mark.skipif(not (activity.cellpose_available() and CELLPOSE_MODEL.exists()),
                                    reason="Cellpose or its model isn't installed here")


def _cli(argv):
    p = argparse.ArgumentParser()
    cli._add_common(p)
    return cli._settings(p.parse_args(argv))


def test_neuron_finder_option_reaches_settings():
    assert _cli(["--mode", "neurons"]).neuron_finder == "built-in"
    assert _cli(["--mode", "neurons", "--neuron-finder", "cellpose"]).neuron_finder == "cellpose"


def test_missing_cellpose_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(activity, "cellpose_available", lambda: False)
    img = {"mean": np.zeros((32, 32)), "valid": np.ones((32, 32), bool)}
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[cellpose\]\""):
        activity.find_neurons_cellpose(img, 8.0)


@needs_cellpose
def test_cellpose_outlines_synthetic_cell_bodies():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:160, 0:160]
    centres = [(30 + 25 * i, 30 + 25 * j) for i in range(5) for j in range(5)]
    mean = rng.normal(100, 3, (160, 160))
    for cy, cx in centres:
        mean += 60 * (np.hypot(yy - cy, xx - cx) < 6)
    valid = np.ones_like(mean, bool)
    lab, info = activity.find_neurons_cellpose({"mean": mean, "valid": valid}, 12.0)
    assert info["neuron_finder"] == "cellpose" and info["cellpose_version"]
    found = lab.max()
    assert 20 <= found <= 30, found  # 25 planted cell bodies
    # numbered 1..N with no gaps
    assert set(np.unique(lab)) == set(range(found + 1))


# ------------------------------------------------------------------ scoring

def _result(tmp_path: Path, xy_px, pixel_size=0.5) -> Path:
    d = tmp_path / "res"
    d.mkdir()
    xy = np.asarray(xy_px, float)
    pd.DataFrame({"neuron_id": range(1, len(xy) + 1), "x_um": xy[:, 0] * pixel_size,
                  "y_um": xy[:, 1] * pixel_size}).to_csv(d / "neurons.csv", index=False)
    (d / "summary.json").write_text(json.dumps({"pixel_size_um": pixel_size,
                                                "settings": {"neuron_finder": "built-in"}}))
    return d


def _regions(tmp_path: Path, centres_xy) -> Path:
    regions = [{"id": i, "coordinates": [[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]}
               for i, (x, y) in enumerate(centres_xy)]
    p = tmp_path / "regions.json"
    p.write_text(json.dumps(regions))
    return p


def test_score_counts_matches_within_distance(tmp_path):
    truth = [(10, 10), (50, 50), (90, 90), (130, 130)]
    found = [(11, 12), (50, 49), (200, 200)]  # two close, one far away; two labelled ones missed
    r = score_neurofinder(_result(tmp_path, found), _regions(tmp_path, truth))
    assert (r["labelled"], r["found"], r["matched"]) == (4, 3, 2)
    assert r["recall"] == pytest.approx(0.5) and r["precision"] == pytest.approx(2 / 3, abs=1e-4)  # saved rounded
    assert r["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3), abs=1e-4)
    assert json.loads((tmp_path / "res" / "score.json").read_text())["matched"] == 2


def test_score_needs_one_to_one_matches(tmp_path):
    # two found neurons next to one labelled neuron count once
    r = score_neurofinder(_result(tmp_path, [(10, 10), (11, 11)]), _regions(tmp_path, [(10, 10)]))
    assert r["matched"] == 1 and r["precision"] == pytest.approx(0.5, abs=1e-4)
