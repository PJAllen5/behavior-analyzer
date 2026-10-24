"""Compare neurite results with the answer key from make_test_neurites.py.

  python tools/check_neurites.py samples/test_neurites results/test_neurites
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

truth_base, res = sys.argv[1], sys.argv[2]
ts = pd.read_csv(f"{truth_base}_somas.csv")
tt = pd.read_csv(f"{truth_base}_truth.csv")
summ = json.load(open(f"{res}/summary.json"))
px = summ.get("pixel_size_um") or 1.0
nf = pd.read_csv(f"{res}/neurite_frames.csv")
nn = pd.read_csv(f"{res}/neurons.csv")
L = "um" if "x_um" in nn else "px"

cost = np.hypot(ts.x.values[:, None] - nn[f"x_{L}"].values[None] / px,
                ts.y.values[:, None] - nn[f"y_{L}"].values[None] / px)
r, c = linear_sum_assignment(cost)
ok = cost[r, c] < 6
match = {int(nn.neuron_id[j]): int(ts.neuron_id[i]) for i, j in zip(r[ok], c[ok])}
print(f"Cell bodies: {len(match)} of {len(ts)} found, {len(nn) - len(match)} extra")

m = nf[nf.neuron_id.isin(match)].assign(truth_id=lambda d: d.neuron_id.map(match))
m = m.merge(tt.rename(columns=lambda c: c if c == "frame" else c + "_t"), left_on=["truth_id", "frame"],
            right_on=["neuron_id_t", "frame"])
err = (m[f"total_length_{L}"] - m.total_length_um_t) / m.total_length_um_t.clip(lower=1)
tot = m.groupby("frame")[[f"total_length_{L}", "total_length_um_t"]].sum()
e_all = (tot.iloc[:, 0] - tot.iloc[:, 1]) / tot.iloc[:, 1]
print(f"Total length, all neurons: median error {e_all.median():+.1%} (first frame {e_all.iloc[0]:+.1%}, "
      f"last {e_all.iloc[-1]:+.1%})")
print(f"Total length per neuron: median error {err.median():+.1%}, "
      f"median absolute error {err.abs().median():.1%}")
err = (m[f"longest_neurite_{L}"] - m.longest_um_t) / m.longest_um_t.clip(lower=1)
print(f"Longest neurite: median error {err.median():+.1%}, median absolute error {err.abs().median():.1%}")
for col in ("primary_neurites", "branch_points", "tips"):
    print(f"{col.replace('_', ' ').capitalize()}: exact in {(m[col] == m[col + '_t']).mean():.0%} of "
          f"neuron-frames, mean {m[col].mean():.2f} vs true {m[col + '_t'].mean():.2f}")
rates = []
for nid, tid in match.items():
    g = tt[tt.neuron_id == tid]
    rt = np.polyfit(g.time_h, g.total_length_um, 1)[0]
    rm = nn.loc[nn.neuron_id == nid, f"growth_rate_{L}_per_h"].iloc[0]
    rates.append((rt, rm))
rates = np.array(rates)
print(f"Growth rate (total length): measured {rates[:, 1].mean():.1f} vs true {rates[:, 0].mean():.1f} "
      f"µm/h on average; per neuron median error "
      f"{np.median(np.abs(rates[:, 1] - rates[:, 0]) / rates[:, 0]):.0%}")
