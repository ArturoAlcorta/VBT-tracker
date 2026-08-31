"""Splits the disk-height curve into eccentric/concentric phases and computes
VBT metrics per phase (average velocity, peak velocity, sticking-point
velocity for the concentric phase).

Ported from the standalone `standalone_scripts/rep_phases.py` script into pure
functions the Celery task can call directly, plus a new sticking-point metric.

Pixel-to-meter scale: a standard olympic plate is 45cm in diameter, so the
mean mask diameter within each phase gives meters-per-pixel for that phase,
without camera calibration. Doing it per phase (not once per video) absorbs
apparent-scale changes (the athlete moving closer/further between sets).

Phase segmentation anchors on the local minima/maxima ("bottoms"/"tops") of
the normalized height curve — via `scipy.signal.find_peaks`, filtered by a
minimum time between reps and a minimum prominence (depth) — rather than on
fixed height thresholds. A rep that doesn't reach some universal "top" or
"bottom" band (shallower squat, athlete starting mid-ROM, camera-to-athlete
distance drifting between sets) is still a real top/bottom, so it's still
detected: each eccentric phase runs top -> bottom, each concentric phase
bottom -> top. A prior version used a 4-state hysteresis machine over fixed
height thresholds (`TOP`/`BOTTOM`) instead; see git history.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks

DISK_DIAMETER_M = 0.45  # standard olympic plate

ECCENTRIC, CONCENTRIC = "eccentric", "concentric"
_TOP, _BOTTOM = "top", "bottom"  # extremum kinds (not height thresholds)

SMOOTH_SECONDS = 0.10  # moving-average window, in time rather than frames
MIN_REP_SECONDS = 0.70  # min spacing between two tops, or two bottoms
MIN_PROMINENCE = 0.15  # min depth of a rep, relative to the full normalized ROM
MIN_PHASE_S = 0.15
MAX_GAP_FRAMES = 5
TRIM_VEL_FRACTION = 0.10  # see _trim_to_motion


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


def _find_extrema(h: np.ndarray, kind: str, distance: int, prominence: float) -> np.ndarray:
    """Indices of local minima (`kind="bottom"`) or maxima (`kind="top"`) in
    normalized height `h`, via scipy's peak finder.

    `find_peaks` can't handle NaN, and long tracking-gap stretches (left NaN
    by `fill_gaps`) shouldn't be allowed to anchor a rep anyway — they're
    masked out to a value that can never win against real data, then any
    "peak" scipy still finds sitting on that masked value is dropped.
    """
    valid = np.isfinite(h)
    if valid.sum() < 3:
        return np.array([], dtype=int)
    unfavorable = np.nanmin(h) - 1 if kind == _TOP else np.nanmax(h) + 1
    signal = np.where(valid, h, unfavorable)
    if kind == _BOTTOM:
        signal = -signal
    peaks, _ = find_peaks(signal, distance=max(1, distance), prominence=prominence)
    return peaks[valid[peaks]]


def _merged_extrema(height: np.ndarray, distance: int, prominence: float) -> list[tuple[int, str]]:
    """All tops and bottoms in `height`, time-ordered and guaranteed to
    alternate. `find_peaks` already keeps same-kind extrema apart via
    `distance`/`prominence`, but tops and bottoms are found independently of
    each other, so two of the same kind can still end up adjacent (e.g. a
    shallow wobble that both qualifies as its own top and doesn't fully
    separate two bottoms) — when that happens, keep only the more extreme
    one of the pair rather than emitting two eccentric or two concentric
    phases in a row.
    """
    tops = [(int(i), _TOP) for i in _find_extrema(height, _TOP, distance, prominence)]
    bottoms = [(int(i), _BOTTOM) for i in _find_extrema(height, _BOTTOM, distance, prominence)]
    extrema = sorted(tops + bottoms, key=lambda e: e[0])

    merged: list[tuple[int, str]] = []
    for idx, kind in extrema:
        if merged and merged[-1][1] == kind:
            prev_idx, _ = merged[-1]
            more_extreme = height[idx] < height[prev_idx] if kind == _BOTTOM else height[idx] > height[prev_idx]
            if more_extreme:
                merged[-1] = (idx, kind)
            continue
        merged.append((idx, kind))
    return merged


def _trim_to_motion(vel_h: np.ndarray, i0: int, i1: int, direction: float, alpha: float) -> tuple[int, int]:
    """Shrinks `[i0, i1]` inward until velocity (signed, in `direction`)
    reaches `alpha` times the segment's own peak velocity, on both ends.

    A `top`/`bottom` extremum anchor is a genuine local max/min of *height*
    by construction, but the lift often doesn't start moving again right
    away — a brief settle at the top, a moment sitting in the hole — so the
    first stretch after the anchor (and, symmetrically, the last stretch
    before the *next* anchor) can be near-motionless. Left untrimmed, that
    dead time gets counted as part of the phase, and — since the anchor
    itself doesn't move — a chart of it visibly bleeds into the plateau
    before the rep. This mirrors the old `_extend_phase`, but shrinks a
    known-good [i0, i1] inward instead of growing an uncertain one outward.
    """
    peak = float(np.nanmax(vel_h[i0 : i1 + 1] * direction))
    if not np.isfinite(peak) or peak <= 0:
        return i0, i1
    threshold = alpha * peak

    a = i0
    while a < i1 and (not np.isfinite(vel_h[a]) or vel_h[a] * direction < threshold):
        a += 1
    b = i1
    while b > a and (not np.isfinite(vel_h[b]) or vel_h[b] * direction < threshold):
        b -= 1
    return a, b


def _sticking_point_velocity(vel_px: np.ndarray, m_per_px: float) -> float | None:
    """Velocity at the concentric sticking point: a genuine local minimum in
    bar speed that the lift recovers from afterward, not just wherever
    velocity happens to be lowest.

    Found via the same peak-finder used for phase segmentation, applied to
    velocity instead of height: a "trough" is a local minimum with enough
    prominence (>=15% of the phase's peak velocity) to be a real dip rather
    than sensor noise. Not every rep has one — a rep that just accelerates
    once and decelerates into lockout, with no mid-lift slowdown-then-push,
    genuinely has no sticking point, and this returns None for it rather
    than forcing a value onto the nearest low point (which used to end up
    being the coast into lockout for long, gradual reps — see git history).
    """
    n = len(vel_px)
    if n < 5:
        return None
    peak = np.nanmax(vel_px)
    if not np.isfinite(peak) or peak <= 0:
        return None
    valid = np.isfinite(vel_px)
    signal = np.where(valid, vel_px, peak + 1)
    troughs, _ = find_peaks(-signal, prominence=0.15 * peak)
    troughs = troughs[valid[troughs]]
    if len(troughs) == 0:
        return None
    deepest = troughs[np.argmin(vel_px[troughs])]
    return float(vel_px[deepest] * m_per_px)


def analyze_phases(
    t: np.ndarray,
    center_y: np.ndarray,
    diameter: np.ndarray,
    disk_diameter_m: float = DISK_DIAMETER_M,
    smooth_seconds: float = SMOOTH_SECONDS,
    min_rep_seconds: float = MIN_REP_SECONDS,
    min_prominence: float = MIN_PROMINENCE,
    min_phase_s: float = MIN_PHASE_S,
    max_gap: int = MAX_GAP_FRAMES,
    trim_vel_fraction: float = TRIM_VEL_FRACTION,
) -> tuple[np.ndarray, list[Phase]]:
    """Returns (normalized height per frame, list of detected phases)."""
    cy_filled = fill_gaps(center_y, max_gap)

    fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 30.0
    smooth_window = max(1, round(smooth_seconds * fps))
    cy_smooth = smooth(cy_filled, smooth_window)

    y_min, y_max = np.nanmin(cy_smooth), np.nanmax(cy_smooth)
    height = (y_max - cy_smooth) / ((y_max - y_min) or 1.0)
    vel_h = np.gradient(height, t)

    distance = max(1, round(min_rep_seconds * fps))
    extrema = _merged_extrema(height, distance, min_prominence)

    phases: list[Phase] = []
    rep = 0
    for (i0, kind0), (i1, kind1) in pairwise(extrema):
        if kind0 == kind1:
            continue  # shouldn't happen after merging, but stay defensive
        kind = ECCENTRIC if kind0 == _TOP else CONCENTRIC
        direction = -1.0 if kind == ECCENTRIC else 1.0
        i0, i1 = _trim_to_motion(vel_h, i0, i1, direction, trim_vel_fraction)
        if kind == CONCENTRIC and rep == 0:
            continue  # concentric before any eccentric: a leading fragment
            # whose start wasn't captured (recording started mid-ascent) —
            # not a real rep, and critically not "rep 1" either, or it'd
            # collide with the rep 1 that's actually about to start
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
            rep=rep, kind=kind,
            t0=float(t[i0]), t1=float(t[i1]), duration=duration,
            v_avg=v_avg, v_peak=v_peak, v_sticking=v_sticking,
        ))

    _estimate_rir(phases)
    return height, phases


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
