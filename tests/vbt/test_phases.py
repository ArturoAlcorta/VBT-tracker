"""Tests for app/vbt/phases.py.

`test_regression_*` runs against a real recorded track.csv (checked in as a
fixture) precisely because the two real bugs found while building this
module — sticking-point velocity picking up the coast into lockout, and a
leading concentric fragment colliding with rep 1's number — only showed up
on real, noisy tracking data. A clean synthetic signal wouldn't have caught
either.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.vbt.phases import (
    CONCENTRIC,
    ECCENTRIC,
    DISK_DIAMETER_M,
    Phase,
    _rir_from_velocity,
    analyze_phases,
    fill_gaps,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_squat_track(
    n_reps: int,
    fps: float = 30.0,
    rep_seconds: float = 3.0,
    y_top: float = 100.0,
    y_bottom: float = 900.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A clean, synthetic multi-rep squat-like track: `n_reps` full
    top->bottom->top cosine cycles, in pixel space (y increases downward,
    matching the app's image-coordinate convention).

    Sampling starts a quarter-cycle before the first top and ends a
    quarter-cycle after the closing top of the last rep, so every real
    extremum falls strictly inside the array — `scipy.signal.find_peaks`
    never reports a peak sitting on the first or last sample, so a track
    that started (or ended) exactly on an extremum would silently lose it.
    """
    margin = rep_seconds / 4
    t = np.arange(-margin, n_reps * rep_seconds + margin, 1.0 / fps)
    height = 0.5 + 0.5 * np.cos(2 * np.pi * t / rep_seconds)  # 1=top, 0=bottom
    t = t - t[0]  # rebase to start at 0, like a real recording would
    if noise_std:
        rng = np.random.default_rng(seed)
        height = height + rng.normal(0, noise_std, size=height.shape)
    center_y = y_top + (1 - height) * (y_bottom - y_top)
    diameter = np.full_like(t, 200.0)
    return t, center_y, diameter


def load_fixture_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, cy, diam = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["time_s"]))
            cy.append(float(row["center_y_px"]) if row["center_y_px"] else np.nan)
            diam.append(float(row["diameter_px"]) if row["diameter_px"] else np.nan)
    return np.array(t), np.array(cy), np.array(diam)


class TestFillGaps:
    def test_interpolates_short_gap(self):
        y = np.array([1.0, 2.0, np.nan, np.nan, 5.0])
        out = fill_gaps(y, max_gap=5)
        assert np.allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_leaves_long_gap_as_nan(self):
        y = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, 5.0])
        out = fill_gaps(y, max_gap=2)
        assert np.isnan(out[2:6]).all()

    def test_leaves_edge_gap_as_nan(self):
        y = np.array([np.nan, np.nan, 3.0, 4.0])
        out = fill_gaps(y, max_gap=5)
        assert np.isnan(out[:2]).all()


class TestAnalyzePhasesSynthetic:
    def test_clean_signal_detects_every_rep(self):
        t, cy, diam = make_squat_track(n_reps=4)
        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        eccentric = [p for p in phases if p.kind == ECCENTRIC]
        concentric = [p for p in phases if p.kind == CONCENTRIC]
        assert len(eccentric) == 4
        assert len(concentric) == 4

    def test_phases_alternate_and_number_sequentially(self):
        t, cy, diam = make_squat_track(n_reps=5)
        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        kinds = [p.kind for p in phases]
        assert kinds == [ECCENTRIC, CONCENTRIC] * 5
        reps = [p.rep for p in phases]
        assert reps == sorted(reps)  # non-decreasing
        assert set(reps) == {1, 2, 3, 4, 5}

    def test_no_duplicate_rep_numbers_across_kind(self):
        # regression guard for the leading-fragment bug: every (rep, kind)
        # pair must be unique — a real bug produced two "rep 1" concentric
        # phases when the recording implicitly started mid-ascent.
        t, cy, diam = make_squat_track(n_reps=6)
        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        keys = [(p.rep, p.kind) for p in phases]
        assert len(keys) == len(set(keys))

    def test_small_top_wobble_does_not_create_extra_reps(self):
        # a small, low-prominence oscillation layered on top of otherwise
        # clean reps (e.g. the athlete swaying slightly while resting at the
        # top) must not be mistaken for its own rep.
        t, cy, diam = make_squat_track(n_reps=3)
        wobble_mask = (np.cos(2 * np.pi * t / 3.0) > 0.9)  # near each top
        cy = cy.copy()
        cy[wobble_mask] += 5.0 * np.sin(2 * np.pi * t[wobble_mask] * 4)  # tiny high-freq jitter
        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        assert sum(1 for p in phases if p.kind == ECCENTRIC) == 3

    def test_recording_starting_mid_ascent_drops_leading_fragment(self):
        t, cy, diam = make_squat_track(n_reps=3)
        # cut the recording so it starts partway up the first concentric
        start = int(0.2 / 3.0 * len(t)) + 5  # a bit past the first bottom
        t, cy, diam = t[start:] - t[start], cy[start:], diam[start:]
        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        reps = [p.rep for p in phases]
        assert 0 not in reps
        assert len(reps) == len(set(zip(reps, [p.kind for p in phases])))

    def test_shallow_rep_still_detected(self):
        # a rep that never reaches the old fixed TOP=0.85/BOTTOM=0.15 band
        # (e.g. fatigue-shortened depth) must still be picked up — this is
        # the whole point of anchoring on local extrema instead. One
        # full-depth rep followed by a much shallower one, built as a single
        # continuous signal (depth switches exactly at a top, where
        # height=1 regardless of which depth is "active", so there's no
        # seam/discontinuity to create a spurious extra extremum).
        fps, rep_seconds, y_top = 30.0, 3.0, 100.0
        depths = [900.0, 400.0]  # full depth, then a shallow rep
        margin = rep_seconds / 4
        t = np.arange(-margin, len(depths) * rep_seconds + margin, 1.0 / fps)
        rep_idx = np.clip((t // rep_seconds).astype(int), 0, len(depths) - 1)
        y_bottom = np.asarray(depths)[rep_idx]
        shape = 0.5 + 0.5 * np.cos(2 * np.pi * t / rep_seconds)  # 1=top, 0=bottom
        cy = y_top + (1 - shape) * (y_bottom - y_top)
        t = t - t[0]
        diam = np.full_like(t, 200.0)

        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        assert sum(1 for p in phases if p.kind == ECCENTRIC) == 2


class TestRirFromVelocity:
    def test_matches_anchors_exactly(self):
        assert _rir_from_velocity(0.93) == pytest.approx(24.7)
        assert _rir_from_velocity(0.135) == pytest.approx(0.0)
        assert _rir_from_velocity(0.39) == pytest.approx(3.9)

    def test_clamps_above_fastest_anchor(self):
        assert _rir_from_velocity(2.0) == pytest.approx(24.7)

    def test_clamps_below_failure_anchor(self):
        assert _rir_from_velocity(0.01) == pytest.approx(0.0)

    def test_interpolates_between_anchors(self):
        rir = _rir_from_velocity(0.5)  # between 0.47 (6.7) and 0.54 (8.8)
        assert 6.7 < rir < 8.8

    def test_monotonic_in_velocity(self):
        velocities = np.linspace(0.1, 1.0, 50)
        rirs = [_rir_from_velocity(v) for v in velocities]
        assert all(a <= b for a, b in zip(rirs, rirs[1:]))


class TestAnalyzePhasesRegression:
    """Guards against the two real bugs found on this exact recording."""

    @pytest.fixture
    def phases(self) -> list[Phase]:
        t, cy, diam = load_fixture_csv(FIXTURES / "jghfvjhgf_track.csv")
        _, phases = analyze_phases(t, cy, diam, DISK_DIAMETER_M)
        return phases

    def test_no_duplicate_rep_kind_pairs(self, phases):
        keys = [(p.rep, p.kind) for p in phases]
        assert len(keys) == len(set(keys))

    def test_reps_are_dense_starting_at_one(self, phases):
        reps = sorted({p.rep for p in phases})
        assert reps == list(range(1, len(reps) + 1))

    def test_concentric_velocities_are_physically_sane(self, phases):
        for p in phases:
            if p.kind != CONCENTRIC:
                continue
            assert 0 < p.v_avg < 3.0  # m/s; a barbell/plate never moves this fast
            if p.v_sticking is not None:
                assert 0 < p.v_sticking <= p.v_peak + 1e-6
            if p.rir is not None:
                assert 0.0 <= p.rir <= 24.7

    def test_sticking_point_none_when_no_genuine_dip(self, phases):
        # most reps in this recording have a single-peaked velocity profile
        # (accelerate once, decelerate into lockout) with no real mid-lift
        # slowdown-then-recovery — the old implementation forced a value
        # onto them anyway (the near-zero lockout velocity); this must not
        # regress.
        concentric = [p for p in phases if p.kind == CONCENTRIC]
        assert any(p.v_sticking is None for p in concentric)
