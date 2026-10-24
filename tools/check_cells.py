"""Compare cell-mode results with the answer key from make_test_cells.py.

  python tools/check_cells.py samples/test_cells results/test_cells

The tests call score() directly and check its numbers.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MATCH_PX = 8


def score(truth_base: str, res: str) -> dict:
    truth = pd.read_csv(f"{truth_base}_truth.csv")
    tdiv = pd.read_csv(f"{truth_base}_divisions.csv")
    tracks = pd.read_csv(f"{res}/tracks.csv")
    lin = pd.read_csv(f"{res}/lineage.csv")
    cells = pd.read_csv(f"{res}/cells.csv")
    summ = json.load(open(f"{res}/summary.json"))

    found = missed = extra = 0
    pair_of = {}  # (frame, truth id) -> track id
    for f, tg in truth.groupby("frame"):
        dg = tracks[tracks.frame == f]
        if dg.empty:
            missed += len(tg)
            continue
        cost = np.hypot(tg.x.values[:, None] - dg.x.values[None], tg.y.values[:, None] - dg.y.values[None])
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < MATCH_PX
        found += ok.sum()
        missed += len(tg) - ok.sum()
        extra += len(dg) - ok.sum()
        for ri, ci in zip(r[ok], c[ok]):
            pair_of[(f, tg.id.values[ri])] = dg.id.values[ci]

    # ID switches: a true cell whose matched track id changes between frames
    switches = 0
    for tid, g in truth.groupby("id"):
        prev = None
        for f in g.frame:
            cur = pair_of.get((f, tid))
            if cur is not None and prev is not None and cur != prev:
                switches += 1
            if cur is not None:
                prev = cur

    # Divisions: found if both true daughters match tracks (within 2 frames) that the
    # results list as daughters of one parent.
    hit = 0
    for d in tdiv.itertuples():
        kids = set()
        for k in (d.daughter_1, d.daughter_2):
            for f in range(d.frame, d.frame + 3):
                if (f, k) in pair_of:
                    kids.add(pair_of[(f, k)])
                    break
        parents = set(lin[lin.daughter_id.isin(kids)].parent_id)
        if len(kids) == 2 and len(parents) == 1 and lin.daughter_id.isin(kids).sum() == 2:
            hit += 1

    # true mean speed, in the same units the results use
    px = summ.get("pixel_size_um") or 1.0
    dt_min = 1 / summ["fps"] / 60
    steps = [s for _, g in truth.groupby("id") for s in np.hypot(np.diff(g.x), np.diff(g.y))]
    L = "um" if summ.get("pixel_size_um") else "px"
    long = cells[cells.frames >= 5]
    return {
        "cells_found_fraction": found / (found + missed),
        "extra_detections": int(extra),
        "frames": int(truth.frame.nunique()),
        "id_changes": int(switches),
        "true_cells": int(truth.id.nunique()),
        "tracks_made": int(tracks.id.nunique()),
        "divisions_true": len(tdiv),
        "divisions_found": hit,
        "divisions_reported": int(lin.parent_id.nunique()),
        "speed_true": float(np.mean(steps) * px / dt_min),
        "speed_measured": float(long[f"mean_speed_{L}_per_min"].median()),
    }


if __name__ == "__main__":
    s = score(sys.argv[1], sys.argv[2])
    print(f"Cell detection: {s['cells_found_fraction']:.1%} of true cells found, "
          f"{s['extra_detections']} extra detections over {s['frames']} frames")
    print(f"Tracking: {s['id_changes']} ID changes (true cells: {s['true_cells']}, tracks made: {s['tracks_made']})")
    print(f"Divisions: {s['divisions_found']} of {s['divisions_true']} true divisions found correctly; "
          f"{s['divisions_reported']} reported ({s['divisions_reported'] - s['divisions_found']} false or partly wrong)")
    print(f"Speed: measured {s['speed_measured']:.3f} vs true {s['speed_true']:.3f} per minute")
