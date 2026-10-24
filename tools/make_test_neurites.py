"""Make a synthetic neurite-outgrowth time-lapse (16-bit TIFF) with a known answer key.

8 fluorescent neurons, 25 frames 10 min apart (4 h), 0.5 µm pixels. Each neuron grows
2-4 neurites at its own rate; some neurites branch partway through and a few pull back.
Also adds debris specks and uneven lighting, which should not be counted.

Writes <out>.tif plus <out>_somas.csv and <out>_truth.csv (per neuron per frame:
total length, longest neurite, primary neurites, branch points, tips).

  python tools/make_test_neurites.py samples/test_neurites
"""
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

W = H = 640
T, DT_MIN, PX_UM = 25, 10.0, 0.5
SOMA_R = 8.0
UP = 4                                   # draw at 4x and shrink, for smooth thin lines
rng = np.random.default_rng(5)
out = sys.argv[1] if len(sys.argv) > 1 else "test_neurites"


def path(x, y, heading, n, wiggle=0.06):
    pts = [(x, y)]
    for _ in range(n):
        heading += rng.normal(0, wiggle)
        x, y = x + np.cos(heading), y + np.sin(heading)
        pts.append((x, y))
    return np.array(pts)


somas = []
while len(somas) < 8:
    x, y = rng.uniform(80, W - 80), rng.uniform(80, H - 80)
    if all(np.hypot(x - a, y - b) > 140 for a, b in somas):
        somas.append((x, y))

neurites = []   # dicts: neuron, pts (px, from soma edge), start frame, rate px/frame, parent, at
for i, (sx, sy) in enumerate(somas, start=1):
    k = rng.integers(2, 5)
    angs = rng.uniform(0, 2 * np.pi) + np.arange(k) * 2 * np.pi / k + rng.normal(0, 0.3, k)
    for a in angs:
        rate = rng.uniform(3, 7)          # px per frame = 9-21 µm/h
        start = rng.integers(0, 4)
        n0 = rng.uniform(8, 20)           # already this long when filming starts
        pts = path(sx + SOMA_R * np.cos(a), sy + SOMA_R * np.sin(a), a, 400)
        retract = rng.random() < 0.15
        neurites.append(dict(neuron=i, pts=pts, start=start, rate=rate, n0=n0, parent=None,
                             retract_at=rng.integers(12, 18) if retract else None))
    # one or two branches off this neuron's neurites
    for _ in range(rng.integers(1, 3)):
        par = [n for n in neurites if n["neuron"] == i and n["parent"] is None]
        p = par[rng.integers(len(par))]
        tb = int(rng.integers(6, 14))
        L_at = p["n0"] + p["rate"] * max(0, tb - p["start"])
        at = int(rng.uniform(0.4, 0.8) * L_at)
        d = p["pts"][min(at + 1, len(p["pts"]) - 1)] - p["pts"][at]
        h = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.7, 1.1)
        bx, by = p["pts"][at]
        neurites.append(dict(neuron=i, pts=path(bx, by, h, 300), start=tb, rate=rng.uniform(2.5, 5),
                             n0=0.0, parent=p, at=at, retract_at=None))


def length_at(n, t):
    if t < n["start"]:
        return n["n0"] if n["parent"] is None else 0.0
    grow = n["n0"] + n["rate"] * (t - n["start"])
    if n["retract_at"] is not None and t > n["retract_at"]:
        peak = n["n0"] + n["rate"] * (n["retract_at"] - n["start"])
        grow = max(4.0, peak - 1.5 * n["rate"] * (t - n["retract_at"]))
    return float(min(grow, len(n["pts"]) - 1))


yy, xx = np.mgrid[0:H, 0:W]
illum = 150 + 120 * (xx / W) + 60 * np.sin(yy / H * np.pi)
debris = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(25)]
frames, truth = [], []
for t in range(T):
    big = np.zeros((H * UP, W * UP), np.float32)
    for n in neurites:
        L = int(length_at(n, t))
        if L < 2:
            continue
        p = (n["pts"][:L + 1] * UP).astype(np.int32)
        cv2.polylines(big, [p], False, 1.0, thickness=int(2 * UP), lineType=cv2.LINE_AA)
        cv2.circle(big, tuple(p[-1]), int(2.2 * UP), 1.0, -1, cv2.LINE_AA)   # growth cone
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA) * 500
    for sx, sy in somas:
        img += 1300 * (np.hypot(xx - sx, yy - sy) < SOMA_R)
    for dx, dy in debris:
        img += 700 * (np.hypot(xx - dx, yy - dy) < 2.0)
    img = ndi.gaussian_filter(img, 0.8) + illum
    img = rng.poisson(img).astype(np.float32) + rng.normal(0, 12, img.shape)
    frames.append(np.clip(img, 0, 65535).astype(np.uint16))
    for i in range(1, len(somas) + 1):
        mine = [n for n in neurites if n["neuron"] == i]
        lens = {id(n): length_at(n, t) for n in mine}
        live = [n for n in mine if lens[id(n)] >= 2]
        prim = [n for n in live if n["parent"] is None]
        br = [n for n in live if n["parent"] is not None and n["at"] < lens[id(n["parent"])]]
        # path length from the cell body to each tip
        far = [lens[id(n)] if n["parent"] is None else n["at"] + lens[id(n)] for n in prim + br]
        truth.append({"frame": t, "time_h": t * DT_MIN / 60, "neuron_id": i,
                      "total_length_um": round(sum(lens[id(n)] for n in prim + br) * PX_UM, 2),
                      "longest_um": round(max(far) * PX_UM if far else 0, 2),
                      "primary_neurites": len(prim), "branch_points": len(br),
                      "tips": len(prim) + len(br)})

tifffile.imwrite(f"{out}.tif", np.stack(frames), imagej=True, resolution=(1 / PX_UM, 1 / PX_UM),
                 metadata={"axes": "TYX", "unit": "um", "finterval": DT_MIN * 60})
pd.DataFrame({"neuron_id": range(1, len(somas) + 1), "x": [s[0] for s in somas],
              "y": [s[1] for s in somas]}).to_csv(f"{out}_somas.csv", index=False)
tr = pd.DataFrame(truth)
tr.to_csv(f"{out}_truth.csv", index=False)
g = tr.groupby("frame").total_length_um.sum()
rate = np.polyfit(tr.groupby("frame").time_h.first(), g / len(somas), 1)[0]
print(f"wrote {out}.tif: {T} frames, {len(somas)} neurons, {len(neurites)} neurites; "
      f"mean length per neuron {g.iloc[0] / len(somas):.0f} -> {g.iloc[-1] / len(somas):.0f} µm "
      f"({rate:.1f} µm/h)")
