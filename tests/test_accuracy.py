"""Each mode, on a generated movie with a known answer key, must stay accurate.

Limits sit a little below today's results, so a change that makes things worse fails.
Today (October 2026): cells 99.4% found, 18/21 divisions; neurons 30/30, 99% of events,
8/8 bursts; neurites 8/8 bodies, length within ~3%; animals: all four behaviors in order.
"""
from __future__ import annotations

import cv2
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def _frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def test_animals_behaviors(analyzed):
    r = analyzed("flock")
    s = r["summary"]
    assert 38 <= s["typical_visible"] <= 42  # 40 birds
    beh = pd.read_csv(r["out"] / "behaviors.csv")
    order = list(beh.behavior)
    # the test video swarms, then flocks, then mills (circles)
    assert order[0] == "disordered"
    assert "collective_motion" in order and "milling" in order
    assert order.index("collective_motion") < order.index("milling")
    mill = beh[beh.behavior == "milling"].iloc[0]
    assert 13 <= mill.start_s <= 17
    assert _frames(r["out"] / "annotated.mp4") > 0


def test_cells(analyzed):
    from check_cells import score

    r = analyzed("cells")
    s = score(str(r["truth"]), str(r["out"]))
    assert s["cells_found_fraction"] >= 0.97
    assert s["extra_detections"] <= 5
    assert s["id_changes"] <= 6
    assert s["divisions_found"] >= 16, f"{s['divisions_found']} of {s['divisions_true']} divisions"
    assert s["divisions_reported"] - s["divisions_found"] <= 3
    assert s["speed_measured"] == pytest.approx(s["speed_true"], rel=0.05)
    assert r["summary"]["pixel_size_um"] == pytest.approx(0.65)  # read from the TIFF
    for f in ("cells.csv", "lineage.csv", "msd.csv", "tracks.csv", "annotated.mp4"):
        assert (r["out"] / f).exists(), f


def test_neuron_activity(analyzed):
    from check_neurons import score

    r = analyzed("neurons")
    s = score(str(r["truth"]), str(r["out"]))
    assert s["firing_found"] >= s["firing_true"] - 2
    assert s["false_neurons"] <= 2
    assert s["events_found"] / s["events_true"] >= 0.95
    assert s["false_events"] / max(s["events_reported"], 1) <= 0.02
    assert s["bursts_found"] == s["bursts_true"]
    assert s["false_bursts"] <= 1
    assert s["max_drift_px"] == pytest.approx(s["max_drift_true"], abs=0.5)
    assert r["summary"]["neurons"]["drift_frames_skipped"] <= 0.01 * r["summary"]["frames_analyzed"]
    assert (r["out"] / "annotated.mp4").exists()


def test_neurite_growth(analyzed):
    from check_neurites import score

    r = analyzed("neurites")
    s = score(str(r["truth"]), str(r["out"]))
    assert s["somas_found"] == s["somas_true"]
    assert s["somas_extra"] <= 1
    assert abs(s["total_length_error"]) <= 0.05
    assert s["per_neuron_length_abs_error"] <= 0.06
    assert s["longest_abs_error"] <= 0.07
    assert s["counts"]["primary_neurites"]["exact"] >= 0.9
    assert s["growth_rate_measured"] == pytest.approx(s["growth_rate_true"], rel=0.05)
    assert (r["out"] / "annotated.mp4").exists()
