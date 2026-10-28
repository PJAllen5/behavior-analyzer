"""Drift correction keeps real drift and ignores estimates it can't trust."""
from __future__ import annotations

import numpy as np

from behavetrack.activity import clean_shifts


def _series(T=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 20.0
    raw = np.column_stack([3 * np.sin(2 * np.pi * t / 70), 2 * np.sin(2 * np.pi * t / 45)])
    raw += rng.normal(0, 0.15, raw.shape)
    return raw, rng.uniform(0.45, 0.7, T)


def test_slow_drift_is_kept():
    raw, resp = _series()
    shifts, good = clean_shifts(raw, resp, diam=12, max_shift=40, fps=20)
    assert good.all()
    np.testing.assert_allclose(shifts, raw)


def test_blank_frames_take_their_neighbours_drift():
    # like Neurofinder 00.00: a run of near-blank frames gives weak matches and wild shifts,
    # some of them with a match strong enough to pass on its own
    raw, resp = _series()
    true = raw.copy()
    bad = np.arange(300, 386)
    rng = np.random.default_rng(1)
    raw[bad] = rng.uniform(-200, 200, (len(bad), 2))
    resp[bad] = rng.uniform(0.1, 0.42, len(bad))
    shifts, good = clean_shifts(raw, resp, diam=12, max_shift=64, fps=7)
    assert not good[bad].any() and good.sum() == len(raw) - len(bad)
    assert np.abs(shifts[bad] - true[bad]).max() < 1.5
    assert np.abs(shifts).max() < 4


def test_a_real_stage_jump_is_followed():
    raw, resp = _series()
    raw[200:] += (15, -10)
    shifts, good = clean_shifts(raw, resp, diam=12, max_shift=64, fps=20)
    assert good[220:].all()
    np.testing.assert_allclose(shifts[250:], raw[250:])
