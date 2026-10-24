"""Compare cell-mode results with the answer key from make_test_cells.py.

  python tools/check_cells.py samples/test_cells results/test_cells
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

truth_base, res = sys.argv[1], sys.argv[2]
truth = pd.read_csv(f"{truth_base}_truth.csv")
tdiv = pd.read_csv(f"{truth_base}_divisions.csv")
tracks = pd.read_csv(f"{res}/tracks.csv")
lin = pd.read_csv(f"{res}/lineage.csv")
MATCH_PX = 8

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

# ID switches: a true cell whose matched track id changes between consecutive frames
switches = 0
for tid, g in truth.groupby("id"):
    prev = None
    for f in g.frame:
        cur = pair_of.get((f, tid))
        if cur is not None and prev is not None and cur != prev:
            switches += 1
        if cur is not None:
            prev = cur

# Divisions: a true division counts as found if a detected division happens within 2 frames
# and its daughters match the true daughters.
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
n_det_div = lin.parent_id.nunique()
print(f"Cell detection: {found / (found + missed):.1%} of true cells found, "
      f"{extra} extra detections over {truth.frame.nunique()} frames")
print(f"Tracking: {switches} ID changes (true cells: {truth.id.nunique()}, tracks made: {tracks.id.nunique()})")
print(f"Divisions: {hit} of {len(tdiv)} true divisions found correctly; "
      f"{n_det_div} reported ({n_det_div - hit} false or partly wrong)")
