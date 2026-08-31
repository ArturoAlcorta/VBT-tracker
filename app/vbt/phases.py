"""Splits the disk-height curve into eccentric/concentric phases and computes
VBT metrics per phase (average velocity, peak velocity, sticking-point
velocity for the concentric phase).

Ported from the standalone `standalone_scripts/rep_phases.py` script into pure
functions the Celery task can call directly, plus a new sticking-point metric.

Pixel-to-meter scale: a standard olympic plate is 45cm in diameter, so the
mean mask diameter within each phase gives meters-per-pixel for that phase,
without camera calibration. Doing it per phase (not once per video) absorbs
apparent-scale changes (the athlete moving closer/further between sets).

Phase segmentation is a 4-state machine over normalized height: up -> going
down -> down -> going up. Thresholds carry hysteresis, so noise around a
threshold doesn't create spurious phases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISK_DIAMETER_M = 0.45  # standard olympic plate

UP, DOWN, ECCENTRIC, CONCENTRIC = "up", "down", "eccentric", "concentric"

TOP = 0.85
BOTTOM = 0.15
VEL_FRACTION = 0.10
MIN_PHASE_S = 0.15
SMOOTH_WINDOW = 5
MAX_GAP_FRAMES = 5


@dataclass
class Phase:
    rep: int
    kind: str  # "eccentric" | "concentric"
    t0: float
    t1: float
    duration: float
    v_avg: float
    v_peak: float
    v_sticking: float | None  # only set for the concentric phase


def fill_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolates gaps of up to max_gap frames; longer ones stay NaN."""
    out = y.copy()
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return out
    idx = np.arange(len(y))
    out[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    i = 0
    while i < len(y):
        if ok[i]:
            i += 1
            continue
        j = i
        while j < len(y) and not ok[j]:
            j += 1
        if j - i > max_gap or i == 0 or j == len(y):
            out[i:j] = np.nan  # long gap or at the edge: don't make up data
        i = j
    return out


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average that ignores NaNs instead of propagating them."""
    if win <= 1:
        return y.copy()
    kernel = np.ones(win) / win
    valid = ~np.isnan(y)
    num = np.convolve(np.where(valid, y, 0.0), kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full(len(y), np.nan)
    np.divide(num, den, out=out, where=den > 0)
    out[~valid] = np.nan
    return out


def _state_series(h: np.ndarray, top: float, bottom: float) -> list[str | None]:
    """4-state machine over normalized height.

    In the middle band the state depends on the last extreme visited, which is
    what gives the hysteresis: without having touched "up" you can't be going
    down.
    """
    states: list[str | None] = []
    last_extreme: str | None = None
    for v in h:
        if np.isnan(v):
            states.append(None)
            continue
        if v >= top:
            last_extreme = UP
            states.append(UP)
        elif v <= bottom:
            last_extreme = DOWN
            states.append(DOWN)
        elif last_extreme == UP:
            states.append("going_down")
        elif last_extreme == DOWN:
            states.append("going_up")
        else:
            states.append(None)  # we don't know where we came from yet
    return states


def _runs_of(states: list[str | None], kind: str) -> list[tuple[int, int]]:
    """Contiguous stretches [start, end) where the state equals `kind`."""
    out, i = [], 0
    while i < len(states):
        if states[i] != kind:
            i += 1
            continue
        j = i
        while j < len(states) and states[j] == kind:
            j += 1
        out.append((i, j))
        i = j
    return out


def _extend_phase(v: np.ndarray, i0: int, i1: int, direction: float, alpha: float) -> tuple[int, int]:
    """Extends a phase outward while the disk keeps moving in the same direction.

    The core of the phase (between thresholds) only covers ~70% of the
    movement. Extending it by velocity recovers the rest without wandering
    into the plateau: it stops as soon as motion falls below alpha times the
    phase's peak or reverses direction, which is exactly the turnaround point.
    Looking for the plateau's maximum doesn't work: if the athlete sways a bit
    while standing, the maximum lands far from the real start of the movement.
    """
    peak = float(np.nanmax(np.abs(v[i0 : i1 + 1])))
    if not np.isfinite(peak) or peak <= 0:
        return i0, i1
    threshold = alpha * peak

    a = i0
    while a > 0 and np.isfinite(v[a - 1]) and v[a - 1] * direction >= threshold:
        a -= 1
    b = i1
    while b < len(v) - 1 and np.isfinite(v[b + 1]) and v[b + 1] * direction >= threshold:
        b += 1
    return a, b


def _sticking_point_velocity(vel_px: np.ndarray, m_per_px: float) -> float | None:
    """Velocity at the concentric sticking point: the local velocity minimum
    once the lift leaves the bottom, before it accelerates back up.

    `_extend_phase` stretches the phase's core into its low-velocity edges, so
    a plain min() over the whole phase would just return the edge value
    instead of the true mid-lift sticking point. Trimming the first/last 10%
    of samples excludes those edges.
    """
    if len(vel_px) < 5:
        return None
    trim = max(1, int(0.1 * len(vel_px)))
    core = vel_px[trim:-trim] if len(vel_px) > 2 * trim else vel_px
    if not np.isfinite(core).any():
        return None
    return float(np.nanmin(core) * m_per_px)


def analyze_phases(
    t: np.ndarray,
    center_y: np.ndarray,
    diameter: np.ndarray,
    disk_diameter_m: float = DISK_DIAMETER_M,
    top: float = TOP,
    bottom: float = BOTTOM,
    vel_fraction: float = VEL_FRACTION,
    min_phase_s: float = MIN_PHASE_S,
    smooth_window: int = SMOOTH_WINDOW,
    max_gap: int = MAX_GAP_FRAMES,
) -> tuple[np.ndarray, list[Phase]]:
    """Returns (normalized height per frame, list of detected phases)."""
    cy_filled = fill_gaps(center_y, max_gap)
    cy_smooth = smooth(cy_filled, smooth_window)

    y_min, y_max = np.nanmin(cy_smooth), np.nanmax(cy_smooth)
    height = (y_max - cy_smooth) / ((y_max - y_min) or 1.0)

    states = _state_series(height, top, bottom)
    vel_h = np.gradient(height, t)

    raw_phases: list[tuple[str, int, int]] = []
    for kind, direction, state_name in ((ECCENTRIC, -1.0, "going_down"), (CONCENTRIC, 1.0, "going_up")):
        for i0, i1 in _runs_of(states, state_name):
            a, b = _extend_phase(vel_h, i0, i1 - 1, direction, vel_fraction)
            raw_phases.append((kind, a, b))
    raw_phases.sort(key=lambda p: p[1])

    phases: list[Phase] = []
    rep = 0
    for kind, i0, i1 in raw_phases:
        duration = float(t[i1] - t[i0])
        if duration < min_phase_s:
            continue

        sl = slice(i0, i1 + 1)
        diam_px = float(np.nanmean(diameter[sl]))
        m_per_px = disk_diameter_m / diam_px if diam_px else float("nan")
        displacement_px = float(abs(cy_smooth[i1] - cy_smooth[i0]))

        dt = np.diff(t[sl])
        vel_px = np.abs(np.diff(cy_smooth[sl])) / np.where(dt > 0, dt, np.nan)
        v_peak = float(np.nanmax(vel_px) * m_per_px) if len(vel_px) else float("nan")
        v_avg = displacement_px * m_per_px / duration if duration > 0 else float("nan")
        v_sticking = _sticking_point_velocity(vel_px, m_per_px) if kind == CONCENTRIC else None

        if kind == ECCENTRIC:
            rep += 1

        phases.append(Phase(
            rep=max(rep, 1), kind=kind,
            t0=float(t[i0]), t1=float(t[i1]), duration=duration,
            v_avg=v_avg, v_peak=v_peak, v_sticking=v_sticking,
        ))

    return height, phases
