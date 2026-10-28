"""Score results against a benchmark's answer key.

  behavetrack score RESULTS --neurofinder path/to/regions.json

Neurofinder: each hand-labelled neuron is matched to at most one found neuron whose centre is
within `max_distance_px` (default 5 pixels, about half a cortical cell body at the usual
two-photon zoom), choosing the pairing with the smallest total distance. Reports:

  recall     share of labelled neurons that were found
  precision  share of found neurons that match a labelled one
  F1         the balance of the two (1 = perfect)

This is a centre-matching score in the spirit of the Neurofinder benchmark, not its official
evaluation code, so compare scores computed here with each other rather than with published ones.
Scores are saved as score.json in the results folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _neuron_centres_px(results_dir: Path) -> np.ndarray:
    nn = pd.read_csv(results_dir / "neurons.csv")
    summary = json.loads((results_dir / "summary.json").read_text())
    if {"x_um", "y_um"} <= set(nn.columns):
        px = summary.get("pixel_size_um") or 1.0
        return nn[["x_um", "y_um"]].to_numpy() / px
    return nn[["x_px", "y_px"]].to_numpy()


def score_neurofinder(results_dir: str | Path, regions_json: str | Path, max_distance_px: float = 5.0) -> dict:
    results_dir = Path(results_dir)
    regions = json.loads(Path(regions_json).read_text())
    truth = np.array([np.mean(r["coordinates"], axis=0)[::-1] for r in regions])  # (x, y)
    found = _neuron_centres_px(results_dir)
    hits, dists = 0, np.array([])
    if len(found) and len(truth):
        cost = np.hypot(truth[:, None, 0] - found[None, :, 0], truth[:, None, 1] - found[None, :, 1])
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < max_distance_px
        hits, dists = int(ok.sum()), cost[r, c][ok]
    recall = hits / len(truth) if len(truth) else 0.0
    precision = hits / len(found) if len(found) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    summary = json.loads((results_dir / "summary.json").read_text())
    out = {
        "benchmark": "neurofinder",
        "answer_key": str(regions_json),
        "labelled": int(len(truth)),
        "found": int(len(found)),
        "matched": hits,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "median_match_distance_px": round(float(np.median(dists)), 2) if len(dists) else None,
        "max_distance_px": max_distance_px,
        "neuron_finder": summary.get("settings", {}).get("neuron_finder", "built-in"),
    }
    (results_dir / "score.json").write_text(json.dumps(out, indent=2))
    return out
