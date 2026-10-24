"""Compare neural-activity results with the answer key from make_test_neurons.py.

  python tools/check_neurons.py samples/test_neurons results/test_neurons
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

truth_base, res = sys.argv[1], sys.argv[2]
tn = pd.read_csv(f"{truth_base}_neurons.csv")
ts = pd.read_csv(f"{truth_base}_spikes.csv")
tb = pd.read_csv(f"{truth_base}_bursts.csv")
summ = json.load(open(f"{res}/summary.json"))
px = summ.get("pixel_size_um") or 1.0
nn = pd.read_csv(f"{res}/neurons.csv")
ev = pd.read_csv(f"{res}/events.csv")
bu = pd.read_csv(f"{res}/bursts.csv")
L = "um" if f"x_um" in nn else "px"
MATCH_PX, MATCH_S = 5.0, 0.35

# --- neurons found (silent ones can only be found with --find-by brightness)
cost = np.hypot(tn.x.values[:, None] - nn[f"x_{L}"].values[None] / px,
                tn.y.values[:, None] - nn[f"y_{L}"].values[None] / px)
r, c = linear_sum_assignment(cost)
ok = cost[r, c] < MATCH_PX
match = {int(tn.neuron_id[i]): int(nn.neuron_id[j]) for i, j in zip(r[ok], c[ok])}
firing = tn[~tn.silent]
found_firing = sum(i in match for i in firing.neuron_id)
found_silent = sum(i in match for i in tn[tn.silent].neuron_id)
print(f"Neurons: {found_firing} of {len(firing)} firing neurons found, {found_silent} of "
      f"{tn.silent.sum()} silent ones; {len(nn) - ok.sum()} false neurons")

# --- events: true spikes closer than 0.35 s merge into one event (the indicator can't separate them)
hit = miss = 0
for tid, g in ts.groupby("neuron_id"):
    times = np.sort(g.time_s.values)
    groups = [times[0]]
    for t in times[1:]:
        if t - groups[-1] > MATCH_S:
            groups.append(t)
    if tid not in match:
        miss += len(groups)
        continue
    det = ev[ev.neuron_id == match[tid]].onset_s.values
    for t in groups:
        if len(det) and np.min(np.abs(det - t)) < MATCH_S:
            hit += 1
        else:
            miss += 1
false = 0
inv = {v: k for k, v in match.items()}
for nid, g in ev.groupby("neuron_id"):
    tt = ts[ts.neuron_id == inv.get(nid, -1)].time_s.values
    for t in g.onset_s.values:
        if not len(tt) or np.min(np.abs(tt - t)) >= MATCH_S:
            false += 1
print(f"Firing events: {hit} of {hit + miss} found ({hit / max(hit + miss, 1):.0%}), "
      f"{false} false events ({false / max(len(ev), 1):.1%} of those reported)")

# --- network bursts
bhit = sum(any(abs(b - t) < 0.5 for t in bu.peak_s) for b in tb.time_s) if len(bu) else 0
print(f"Network bursts: {bhit} of {len(tb)} found, {len(bu) - bhit} false")
print(f"Largest drift corrected: {summ['neurons'].get('max_drift_px')} px")
