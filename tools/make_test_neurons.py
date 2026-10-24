"""Make a synthetic calcium-imaging movie (16-bit TIFF) with a known answer key.

40 neurons (a quarter of them silent), 20 frames/s for 90 s, 0.8 µm pixels.
Neurons fire on their own and also join network bursts every 8-15 s. The movie has
the things real recordings have: neuropil haze that brightens during bursts,
photobleaching, slow stage drift, and camera noise.

Writes <out>.tif plus <out>_neurons.csv, <out>_spikes.csv and <out>_bursts.csv.

  python tools/make_test_neurons.py samples/test_neurons
"""
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

W = H = 320
FPS, SECONDS, PX_UM = 20.0, 90.0, 0.8
T = int(FPS * SECONDS)
N, R = 40, 6.0                      # neurons, soma radius (px)
TAU_RISE, TAU_DECAY = 0.05, 0.6     # seconds (GCaMP-like)
rng = np.random.default_rng(11)
out = sys.argv[1] if len(sys.argv) > 1 else "test_neurons"

# --- where the neurons are
pos = []
while len(pos) < N:
    x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
    if all(np.hypot(x - a, y - b) > 3.2 * R for a, b in pos):
        pos.append((x, y))
silent = np.zeros(N, bool)
silent[rng.choice(N, N // 4, replace=False)] = True
base = rng.uniform(150, 400, N)     # resting brightness above the neuropil

yy, xx = np.mgrid[0:H, 0:W]
foot = np.zeros((N, H * W), np.float32)
for i, (x, y) in enumerate(pos):
    r = np.hypot(xx - x, yy - y)
    f = (r < R).astype(np.float32)
    f[r < 0.45 * R] = 0.5           # GCaMP is dimmer over the nucleus
    foot[i] = ndi.gaussian_filter(f, 0.8).ravel()

# --- spikes: independent firing + network bursts
t = np.arange(T) / FPS
spikes = []
for i in range(N):
    if silent[i]:
        continue
    rate = rng.uniform(0.02, 0.25)
    tt = 0.0
    while True:
        tt += rng.exponential(1 / rate)
        if tt >= SECONDS - 1:
            break
        spikes.append((i + 1, tt, rng.uniform(0.4, 1.0), False))
bursts = []
tb = rng.uniform(4, 8)
active = np.flatnonzero(~silent)
while tb < SECONDS - 2:
    who = active[rng.random(len(active)) < 0.7]
    bursts.append({"time_s": round(tb, 3), "neurons_recruited": len(who)})
    for i in who:
        spikes.append((i + 1, tb + rng.normal(0, 0.05), rng.uniform(0.6, 1.4), True))
    tb += rng.uniform(8, 15)
sp = pd.DataFrame(spikes, columns=["neuron_id", "time_s", "amplitude_dff", "in_burst"]).sort_values(["neuron_id", "time_s"])

kt = np.arange(0, 6 * TAU_DECAY, 1 / FPS)
kern = (1 - np.exp(-kt / TAU_RISE)) * np.exp(-kt / TAU_DECAY)
kern /= kern.max()
dff = np.zeros((T, N), np.float32)
for r in sp.itertuples():
    f0 = int(round(r.time_s * FPS))
    n = min(len(kern), T - f0)
    if n > 0 and f0 >= 0:
        dff[f0:f0 + n, r.neuron_id - 1] += r.amplitude_dff * kern[:n]

# neuropil: smooth haze that brightens a little during network bursts
neuropil = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 25)
neuropil = (90 + 40 * (neuropil - neuropil.min()) / np.ptp(neuropil)).astype(np.float32).ravel()
np_act = np.zeros(T, np.float32)
for b in bursts:
    f0 = int(b["time_s"] * FPS)
    n = min(len(kern), T - f0)
    np_act[f0:f0 + n] += 0.25 * kern[:n]

bleach = np.exp(-t / 400.0).astype(np.float32)
dx = 3.0 * np.sin(2 * np.pi * t / 70.0) + rng.normal(0, 0.15, T)
dy = 2.0 * np.sin(2 * np.pi * t / 45.0 + 1) + rng.normal(0, 0.15, T)

frames = np.zeros((T, H, W), np.uint16)
for k in range(T):
    img = neuropil * (1 + np_act[k]) + (base * (1 + dff[k])) @ foot
    img = (img * bleach[k]).reshape(H, W)
    M = np.float32([[1, 0, dx[k]], [0, 1, dy[k]]])
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    img = rng.poisson(np.maximum(img, 0)).astype(np.float32) + 100 + rng.normal(0, 5, img.shape)
    frames[k] = np.clip(img, 0, 65535).astype(np.uint16)

tifffile.imwrite(f"{out}.tif", frames, imagej=True, resolution=(1 / PX_UM, 1 / PX_UM),
                 metadata={"axes": "TYX", "unit": "um", "finterval": 1 / FPS})
pd.DataFrame({"neuron_id": np.arange(1, N + 1), "x": [p[0] for p in pos], "y": [p[1] for p in pos],
              "silent": silent, "baseline": base.round(1)}).to_csv(f"{out}_neurons.csv", index=False)
sp.round(4).to_csv(f"{out}_spikes.csv", index=False)
pd.DataFrame(bursts).to_csv(f"{out}_bursts.csv", index=False)
print(f"wrote {out}.tif: {T} frames at {FPS:g}/s, {N} neurons ({silent.sum()} silent), "
      f"{len(sp)} firing events, {len(bursts)} network bursts")
