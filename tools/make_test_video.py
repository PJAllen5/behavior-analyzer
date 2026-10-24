"""Make a synthetic test video with known behaviors, to check the analyzer.

  0-5 s    disordered (random swarming)
  5-11 s   collective_motion (flock travelling together)
  11-17 s  milling (circling a centre)
  17-20 s  resting
"""
import sys

import cv2
import numpy as np

W, H, FPS, N = 800, 600, 30, 40
PHASES = [(0, 5, "disordered"), (5, 11, "collective"), (11, 17, "milling"), (17, 20, "resting")]
rng = np.random.default_rng(1)

pos = rng.uniform([100, 100], [W - 100, H - 100], (N, 2))
ang = rng.uniform(-np.pi, np.pi, N)
speed = np.full(N, 90.0)
flock_dir = 0.0
mill_speeds = rng.uniform(40, 180, N)  # different speeds spread birds around the ring
out = sys.argv[1] if len(sys.argv) > 1 else "test_flock.mp4"
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
dt = 1 / FPS
centre = np.array([W / 2, H / 2])

for f in range(PHASES[-1][1] * FPS):
    t = f / FPS
    phase = next(p for a, b, p in PHASES if a <= t < b)
    if phase == "disordered":
        ang += rng.normal(0, 0.35, N)
        target_speed = 90
    elif phase == "collective":
        # whole group steers toward a shared direction, gently curving to stay on screen
        flock_dir += 0.5 * dt
        to_c = centre - pos.mean(0)
        desired = flock_dir + 0.002 * (np.cos(flock_dir) * to_c[1] - np.sin(flock_dir) * to_c[0])
        cohesion = np.arctan2(*(pos.mean(0) - pos).T[::-1])
        d = np.linalg.norm(pos.mean(0) - pos, axis=1)
        mix = np.clip((d - 150) / 150, 0, 0.4)
        tgt = np.arctan2((1 - mix) * np.sin(desired) + mix * np.sin(cohesion),
                         (1 - mix) * np.cos(desired) + mix * np.cos(cohesion))
        ang += 0.15 * np.angle(np.exp(1j * (tgt - ang))) + rng.normal(0, 0.05, N)
        target_speed = 110
    elif phase == "milling":
        r = pos - centre
        rad = np.linalg.norm(r, axis=1)
        tang = np.arctan2(r[:, 0], -r[:, 1])  # counter-clockwise tangent
        inward = np.arctan2(-r[:, 1], -r[:, 0])
        k = np.clip((rad - 170) / 100, -0.5, 0.5)
        tgt = np.angle((1 - abs(k)) * np.exp(1j * tang) + k * np.exp(1j * inward))
        ang += 0.2 * np.angle(np.exp(1j * (tgt - ang))) + rng.normal(0, 0.05, N)
        target_speed = mill_speeds
    else:
        target_speed = 0
    if np.any(target_speed):
        # separation: turn away from anyone closer than 30 px (keeps birds from merging)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = (diff ** 2).sum(-1) + np.eye(N) * 1e9
        push = (diff / d2[..., None] * (d2 < 30 ** 2)[..., None]).sum(1)
        strong = np.linalg.norm(push, axis=1) > 1e-3
        away = np.arctan2(push[:, 1], push[:, 0])
        ang[strong] += 0.3 * np.angle(np.exp(1j * (away[strong] - ang[strong])))
    speed += (target_speed - speed) * 0.1
    # soft walls
    for ax, lim in ((0, W), (1, H)):
        lo, hi = pos[:, ax] < 40, pos[:, ax] > lim - 40
        if ax == 0:
            ang[lo | hi] = np.pi - ang[lo | hi]
        else:
            ang[lo | hi] = -ang[lo | hi]
        pos[:, ax] = np.clip(pos[:, ax], 40, lim - 40)
    v = speed[:, None] * np.c_[np.cos(ang), np.sin(ang)]
    pos += v * dt

    img = np.full((H, W, 3), (205, 225, 235), np.uint8)
    img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))  # sensor noise
    for (x, y), a in zip(pos, ang):
        cv2.ellipse(img, (int(x), int(y)), (7, 3), np.degrees(a), 0, 360, (40, 40, 40), -1)
    wr.write(img)
wr.release()
print("wrote", out)
