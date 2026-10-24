"""Make a synthetic microscope time-lapse (16-bit TIFF) with a known answer key.

25 fluorescent cells, 72 frames, 5 min apart (6 h), 0.65 µm pixels.
Cells crawl with a drift toward the right (+x), and divide.
Writes <out>.tif plus <out>_truth.csv (positions) and <out>_divisions.csv.

  python tools/make_test_cells.py samples/test_cells
"""
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

W = H = 512
T, DT_MIN, PX_UM = 72, 5.0, 0.65
SPEED_PX = 3.0           # px per frame  (~0.39 µm/min)
DRIFT_PX = 0.8           # px per frame toward +x
R_ADULT = 10.0
rng = np.random.default_rng(7)
out = sys.argv[1] if len(sys.argv) > 1 else "test_cells"

cells = []  # dicts: id, x, y, ang, r, age (frames), cycle (frames)
next_id = 1
while len(cells) < 25:
    x, y = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
    if all(np.hypot(x - c["x"], y - c["y"]) > 3 * R_ADULT for c in cells):
        cells.append(dict(id=next_id, x=x, y=y, ang=rng.uniform(-np.pi, np.pi), r=R_ADULT,
                          age=rng.uniform(0, 80), cycle=rng.uniform(60, 110)))
        next_id += 1

truth, divisions, frames = [], [], []
yy, xx = np.mgrid[0:H, 0:W]
illum = 200 + 220 * (xx / W) + 80 * (yy / H)  # uneven illumination
for t in range(T):
    # division
    new = []
    for c in cells:
        if c["age"] >= c["cycle"]:
            a = rng.uniform(0, np.pi)
            kids = []
            for s in (-1, 1):
                kids.append(dict(id=next_id, x=c["x"] + s * 0.8 * c["r"] * np.cos(a),
                                 y=c["y"] + s * 0.8 * c["r"] * np.sin(a), ang=a + (s < 0) * np.pi,
                                 r=c["r"] / np.sqrt(2) * 1.05, age=0, cycle=rng.uniform(70, 110)))
                next_id += 1
            divisions.append({"frame": t, "parent": c["id"], "daughter_1": kids[0]["id"],
                              "daughter_2": kids[1]["id"]})
            new += kids
        else:
            new.append(c)
    cells = new
    # movement: persistent random walk + drift + gentle repulsion
    for c in cells:
        c["ang"] += rng.normal(0, 0.35)
        vx = SPEED_PX * np.cos(c["ang"]) + DRIFT_PX
        vy = SPEED_PX * np.sin(c["ang"])
        for o in cells:
            if o is c:
                continue
            dx, dy = c["x"] - o["x"], c["y"] - o["y"]
            d = np.hypot(dx, dy)
            if 0 < d < 2.2 * R_ADULT:
                vx += 2.0 * dx / d
                vy += 2.0 * dy / d
        c["x"] = float(np.clip(c["x"] + vx, 15, W - 15))
        c["y"] = float(np.clip(c["y"] + vy, 15, H - 15))
        if c["x"] in (15, W - 15):
            c["ang"] = np.pi - c["ang"]
        if c["y"] in (15, H - 15):
            c["ang"] = -c["ang"]
        c["r"] = min(R_ADULT, c["r"] + 0.05)
        c["age"] += 1
        truth.append({"frame": t, "id": c["id"], "x": c["x"], "y": c["y"]})
    # draw
    img = np.zeros((H, W), np.float32)
    for c in cells:
        img += 900 * (np.hypot(xx - c["x"], yy - c["y"]) < c["r"])
    img = ndi.gaussian_filter(img, 1.5) + illum
    img = rng.poisson(img).astype(np.float32) + rng.normal(0, 15, img.shape)
    frames.append(np.clip(img, 0, 65535).astype(np.uint16))

tifffile.imwrite(f"{out}.tif", np.stack(frames), imagej=True,
                 resolution=(1 / PX_UM, 1 / PX_UM),
                 metadata={"axes": "TYX", "unit": "um", "finterval": DT_MIN * 60})
tr = pd.DataFrame(truth)
tr.to_csv(f"{out}_truth.csv", index=False)
pd.DataFrame(divisions).to_csv(f"{out}_divisions.csv", index=False)

# answer key
steps = []
for _, g in tr.groupby("id"):
    d = np.hypot(np.diff(g.x), np.diff(g.y))
    steps += list(d)
print(f"wrote {out}.tif: {T} frames, cells {tr[tr.frame == 0].shape[0]} -> {tr[tr.frame == T - 1].shape[0]}, "
      f"{len(divisions)} divisions, true mean speed {np.mean(steps) * PX_UM / DT_MIN:.3f} µm/min")
