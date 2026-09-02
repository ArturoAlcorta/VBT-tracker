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
from itertools import pairwise

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
    rir: float | None = None  # estimated reps in reserve, concentric phases only


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
    a plain min() over the whole phase would just return an edge value
    instead of the true mid-lift sticking point. A *fixed* 10% trim off each
    end (the original approach) isn't enough to exclude that edge on a rep
    whose coast into lockout is long and gradual — it silently reports the
    near-zero velocity right before the disk stops, not a real sticking
    point. Instead, trim the start (still a fixed 10%, since that edge is
    short and near-constant) and cut the end at the last point the rep is
    still moving at >=25% of its peak velocity — everything after that is
    deceleration into lockout, not a candidate for the sticking point,
    regardless of how long it drags on for.
    """
    n = len(vel_px)
    if n < 5:
        return None
    peak = np.nanmax(vel_px)
    if not np.isfinite(peak) or peak <= 0:
        return None
    above_threshold = np.flatnonzero(np.isfinite(vel_px) & (vel_px >= 0.25 * peak))
    if len(above_threshold) == 0:
        return None
    start = max(1, int(0.1 * n))
    end = above_threshold[-1] + 1
    core = vel_px[start:end]
    if len(core) == 0 or not np.isfinite(core).any():
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

    phases = _drop_incomplete_reps(phases)
    _estimate_rir(phases)
    return height, phases


def _drop_incomplete_reps(phases: list[Phase]) -> list[Phase]:
    """Drops any rep that doesn't have both an eccentric and a concentric
    phase — typically the last rep in the video, cut off mid-ascent before
    a concentric phase could be detected (rarer: a leading fragment cut off
    mid-descent). Real data, but useless shown as half a rep, and it was
    confusing rendered as a row with dashes for the missing half."""
    kinds_by_rep: dict[int, set[str]] = {}
    for p in phases:
        kinds_by_rep.setdefault(p.rep, set()).add(p.kind)
    complete_reps = {rep for rep, kinds in kinds_by_rep.items() if {ECCENTRIC, CONCENTRIC} <= kinds}
    return [p for p in phases if p.rep in complete_reps]


# --- RIR (reps in reserve) estimation ---------------------------------------
#
# Fixed velocity -> RIR thresholds: a rep's own absolute velocity maps
# directly to an RIR estimate, independent of any other rep in the set. We
# tried comparing each rep to the set's fastest rep instead (see git
# history), but that reports a high RIR for whichever rep happens to be
# fastest even when every rep in the set is objectively slow — i.e. it
# measures relative fatigue within the set, not actual proximity to failure.
# Fixed thresholds fix that at the cost of needing a per-exercise velocity
# scale, which for now is bench-press-only (see caveats below).
#
# Anchors come from two tables in the same source, same 22-24 man cohort:
#
#   González-Badillo JJ, Yañez-García JM, Mora-Custodio R, Rodríguez-Rosell D.
#   "Velocity Loss as a Variable for Monitoring Resistance Exercise."
#   Int J Sports Med. 2017;38(3):217-225. doi:10.1055/s-0042-120324
#
# 1) %1RM -> first-rep MPV (their %1RM-velocity classification table, from
#    González-Badillo & Sánchez-Medina 2010): tells us the effective %1RM a
#    given velocity represents, whatever the actual prescribed load was —
#    the whole premise of VBT is that velocity reflects current effort
#    regardless of load.
# 2) %1RM -> mean reps to failure (their Table 1): at that effective %1RM, a
#    fresh set would produce N reps; RIR = N - 1 (the current rep is the
#    first of those N). 100% 1RM is anchored separately using the paper's own
#    reported failure/1RM MPV (~0.12-0.15 m/s, midpoint 0.135) since Table 1
#    doesn't include a literal 100% row — by definition 1 rep possible, 0 RIR.
#
# We feed each rep's `v_avg` in as the velocity — the closest thing we
# compute to the paper's MPV (mean *propulsive* velocity, i.e. only the
# accelerating portion of the concentric phase, acceleration >= -9.81 m/s²).
# `v_avg` is displacement/duration over the *whole* concentric phase
# (including the deceleration into lockout, since we don't segment out a
# propulsive sub-phase), so it runs a bit low relative to true MPV — still
# a much closer match than the sticking-point velocity we used originally,
# which the literature doesn't use as a monitoring metric at all (it's a
# different line of research — locating the weakest point in the strength
# curve, e.g. Kompf & Arandjelović's sticking-point review — not fatigue/RIR
# estimation). The whole thing is bench-press-specific (the equivalent
# squat/deadlift study, Rodríguez-Rosell et al. 2020, J Strength Cond Res
# 34(9):2537-2547, is paywalled) — applied to every exercise regardless.
# Treat this as a rough, directional estimate, not a precise figure.

# (velocity m/s, RIR) anchors, fastest to slowest — see derivation above.
_BP_VELOCITY_RIR_ANCHORS = (
    (0.93, 24.7), (0.86, 21.7), (0.79, 18.6), (0.71, 15.2),
    (0.62, 11.6), (0.54, 8.8), (0.47, 6.7), (0.39, 3.9), (0.135, 0.0),
)


def _rir_from_velocity(v: float) -> float:
    """Piecewise-linear interpolation over `_BP_VELOCITY_RIR_ANCHORS`; clamps
    to the table's ends rather than extrapolating past them."""
    anchors = _BP_VELOCITY_RIR_ANCHORS
    if v >= anchors[0][0]:
        return anchors[0][1]
    if v <= anchors[-1][0]:
        return anchors[-1][1]
    for (v_hi, rir_hi), (v_lo, rir_lo) in pairwise(anchors):
        if v_lo <= v <= v_hi:
            frac = (v - v_lo) / (v_hi - v_lo)
            return rir_lo + frac * (rir_hi - rir_lo)
    return anchors[-1][1]  # unreachable


def _estimate_rir(phases: list[Phase]) -> None:
    """Sets `.rir` in place on every concentric phase, from that rep's own
    mean velocity alone — no comparison to other reps in the set."""
    for phase in phases:
        if phase.kind == CONCENTRIC and np.isfinite(phase.v_avg):
            phase.rir = _rir_from_velocity(phase.v_avg)
