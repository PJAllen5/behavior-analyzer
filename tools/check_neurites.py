"""Compare neurite results with the answer key from make_test_neurites.py.

  python tools/check_neurites.py samples/test_neurites results/test_neurites

The tests call score() directly and check its numbers.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def score(truth_base: str, res: str) -> dict:
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

    m = nf[nf.neuron_id.isin(match)].assign(truth_id=lambda d: d.neuron_id.map(match))
    m = m.merge(tt.rename(columns=lambda c: c if c == "frame" else c + "_t"), left_on=["truth_id", "frame"],
                right_on=["neuron_id_t", "frame"])
    err = (m[f"total_length_{L}"] - m.total_length_um_t) / m.total_length_um_t.clip(lower=1)
    tot = m.groupby("frame")[[f"total_length_{L}", "total_length_um_t"]].sum()
    e_all = (tot.iloc[:, 0] - tot.iloc[:, 1]) / tot.iloc[:, 1]
    err_long = (m[f"longest_neurite_{L}"] - m.longest_um_t) / m.longest_um_t.clip(lower=1)
    counts = {col: {"exact": float((m[col] == m[col + "_t"]).mean()), "mean": float(m[col].mean()),
                    "true_mean": float(m[col + "_t"].mean())}
              for col in ("primary_neurites", "branch_points", "tips")}
    rates = []
    for nid, tid in match.items():
        g = tt[tt.neuron_id == tid]
        rt = np.polyfit(g.time_h, g.total_length_um, 1)[0]
        rm = nn.loc[nn.neuron_id == nid, f"growth_rate_{L}_per_h"].iloc[0]
        rates.append((rt, rm))
    rates = np.array(rates)
    return {
        "somas_found": len(match),
        "somas_true": len(ts),
        "somas_extra": int(len(nn) - len(match)),
        "total_length_error": float(e_all.median()),
        "total_length_error_first": float(e_all.iloc[0]),
        "total_length_error_last": float(e_all.iloc[-1]),
        "per_neuron_length_error": float(err.median()),
        "per_neuron_length_abs_error": float(err.abs().median()),
        "longest_error": float(err_long.median()),
        "longest_abs_error": float(err_long.abs().median()),
        "counts": counts,
        "growth_rate_measured": float(rates[:, 1].mean()),
        "growth_rate_true": float(rates[:, 0].mean()),
        "growth_rate_per_neuron_error": float(np.median(np.abs(rates[:, 1] - rates[:, 0]) / rates[:, 0])),
    }


if __name__ == "__main__":
    s = score(sys.argv[1], sys.argv[2])
    print(f"Cell bodies: {s['somas_found']} of {s['somas_true']} found, {s['somas_extra']} extra")
    print(f"Total length, all neurons: median error {s['total_length_error']:+.1%} (first frame "
          f"{s['total_length_error_first']:+.1%}, last {s['total_length_error_last']:+.1%})")
    print(f"Total length per neuron: median error {s['per_neuron_length_error']:+.1%}, "
          f"median absolute error {s['per_neuron_length_abs_error']:.1%}")
    print(f"Longest neurite: median error {s['longest_error']:+.1%}, "
          f"median absolute error {s['longest_abs_error']:.1%}")
    for col, c in s["counts"].items():
        print(f"{col.replace('_', ' ').capitalize()}: exact in {c['exact']:.0%} of "
              f"neuron-frames, mean {c['mean']:.2f} vs true {c['true_mean']:.2f}")
    print(f"Growth rate (total length): measured {s['growth_rate_measured']:.1f} vs true "
          f"{s['growth_rate_true']:.1f} µm/h on average; per neuron median error "
          f"{s['growth_rate_per_neuron_error']:.0%}")
