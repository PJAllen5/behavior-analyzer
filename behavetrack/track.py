"""Multi-object tracker: links detections across frames so each object keeps one ID.

Uses constant-velocity prediction + optimal (Hungarian) matching on distance.
Optionally detects cell division: one object becoming two nearby, smaller objects.
The parent's ID ends and each daughter gets a new ID that remembers its parent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

EXTRA_COLS = ["area_px", "intensity", "eccentricity", "diameter_px"]


@dataclass
class _Track:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    size: float
    area: float
    cls: str
    last_frame: int
    parent: int | None = None
    hits: int = 1
    pending: list = field(default_factory=list)  # rows held until track is confirmed


class Tracker:
    def __init__(self, max_dist: float | None = None, max_missed: int = 10, min_hits: int = 3,
                 gate_sizes: float = 4.0, divisions: bool = False):
        """max_dist: furthest (pixels) an object may jump between frames. None = auto
        (gate_sizes x the object's size). max_missed: frames an object may vanish before
        its ID ends. min_hits: detections needed before a track counts (filters flicker).
        divisions: look for one object splitting into two (cell division)."""
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.gate_sizes = gate_sizes
        self.divisions = divisions
        self.tracks: dict[int, _Track] = {}
        self.next_id = 1
        self.rows: list[tuple] = []  # frame, id, x, y, w, h, cls, conf, *extras
        self.lineage: dict[int, dict] = {}  # id -> parent, division frame

    @staticmethod
    def _area(d) -> float:
        return d[6]["area_px"] if len(d) > 6 and "area_px" in d[6] else d[2] * d[3]

    def _gate(self, t: _Track, gap: int) -> float:
        return (self.max_dist or max(self.gate_sizes * t.size, 15.0)) * gap

    def update(self, frame_idx: int, dets: list) -> None:
        ids = list(self.tracks)
        matches: dict[int, int] = {}  # track id -> detection index
        if ids and dets:
            pred = np.array([self.tracks[i].pos + self.tracks[i].vel * (frame_idx - self.tracks[i].last_frame)
                             for i in ids])
            dpos = np.array([[d[0], d[1]] for d in dets])
            cost = np.linalg.norm(pred[:, None, :] - dpos[None, :, :], axis=2)
            for ti, tid in enumerate(ids):
                t = self.tracks[tid]
                gate = self._gate(t, frame_idx - t.last_frame)
                for di, d in enumerate(dets):
                    if cost[ti, di] > gate or d[4] != t.cls:
                        cost[ti, di] = 1e9
            r, c = linear_sum_assignment(cost)
            matches = {ids[ti]: di for ti, di in zip(r, c) if cost[ti, di] < 1e9}

        used = set(matches.values())
        daughters: dict[int, int] = {}  # detection index -> parent id
        if self.divisions and dets:
            for u, du in enumerate(dets):
                if u in used:
                    continue
                best = None
                for tid, di in matches.items():
                    t = self.tracks[tid]
                    if t.last_frame != frame_idx - 1 or t.hits < self.min_hits:
                        continue
                    dm = dets[di]
                    a1, a2 = self._area(du), self._area(dm)
                    gate = 1.5 * t.size
                    d1 = np.hypot(du[0] - t.pos[0], du[1] - t.pos[1])
                    d2 = np.hypot(dm[0] - t.pos[0], dm[1] - t.pos[1])
                    if (d1 < gate and d2 < gate and max(a1, a2) < 0.85 * t.area
                            and 0.6 * t.area < a1 + a2 < 1.7 * t.area):
                        score = d1 + d2
                        if best is None or score < best[0]:
                            best = (score, tid, di)
                if best:
                    _, tid, di = best
                    del matches[tid]
                    daughters[u] = daughters[di] = tid
                    self.lineage.setdefault(tid, {"parent": self.tracks[tid].parent})
                    self.lineage[tid]["divided_at"] = frame_idx
                    del self.tracks[tid]
                    used.add(u)

        for tid, di in matches.items():
            t = self.tracks[tid]
            d = dets[di]
            new = np.array([d[0], d[1]])
            gap = frame_idx - t.last_frame
            t.vel = 0.6 * t.vel + 0.4 * (new - t.pos) / gap
            t.pos = new
            t.size = 0.8 * t.size + 0.2 * max(d[2], d[3])
            t.area = 0.7 * t.area + 0.3 * self._area(d)
            t.last_frame = frame_idx
            t.hits += 1
            self._emit(t, frame_idx, d)

        matched_dets = set(matches.values())
        for di, d in enumerate(dets):
            if di in matched_dets:
                continue
            parent = daughters.get(di)
            t = _Track(self.next_id, np.array([d[0], d[1]]), np.zeros(2),
                       max(d[2], d[3]), self._area(d), d[4], frame_idx, parent=parent)
            if parent is not None:
                t.hits = self.min_hits  # daughters count straight away
                self.lineage[t.id] = {"parent": parent, "born_at": frame_idx}
            self.next_id += 1
            self.tracks[t.id] = t
            self._emit(t, frame_idx, d)

        for tid in list(self.tracks):
            if frame_idx - self.tracks[tid].last_frame > self.max_missed:
                del self.tracks[tid]

    def _emit(self, t: _Track, frame_idx: int, d):
        extra = d[6] if len(d) > 6 else {}
        row = (frame_idx, t.id, d[0], d[1], d[2], d[3], d[4], d[5],
               *(extra.get(k, np.nan) for k in EXTRA_COLS))
        if t.hits < self.min_hits:
            t.pending.append(row)
            return
        if t.pending:
            self.rows.extend(t.pending)
            t.pending = []
        self.rows.append(row)
