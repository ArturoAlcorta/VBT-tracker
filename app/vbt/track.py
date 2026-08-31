"""Disk tracking: runs the YOLO-seg model over a video and records the disk's
vertical center position and diameter per frame.

Ported from the standalone `standalone_scripts/track_height.py` script into
pure functions the Celery task can call directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TrackResult:
    fps: float
    t: np.ndarray
    center_y: np.ndarray  # px, NaN where the disk wasn't detected
    diameter: np.ndarray  # area-equivalent diameter, px; NaN where not detected


def _circularity(pts: np.ndarray) -> float:
    """4*pi*area / perimeter^2: 1.0 is a perfect circle, ~0.2 a blob."""
    p = pts.astype(np.float32)
    perimeter = cv2.arcLength(p, True)
    return float(4.0 * np.pi * cv2.contourArea(p) / perimeter**2) if perimeter else 0.0


def track_disk(
    video_path: Path,
    weights_path: Path,
    conf: float,
    device: str,
    min_circularity: float = 0.5,
) -> TrackResult:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    model = YOLO(str(weights_path))
    # the one2one (NMS-free) head spreads confidence across neighboring anchors
    # and drops frames below threshold; one2many + NMS gives ~0.96 conf, 1 det/frame
    model.model.model[-1].end2end = False
    results = model.track(
        source=str(video_path), stream=True, conf=conf,
        device=device, verbose=False, persist=True,
    )

    center_y: list[float] = []
    diameter: list[float] = []
    for res in results:
        cy, diam = np.nan, np.nan
        if res.masks is not None and len(res.masks) > 0:
            polys = res.masks.xy  # list of (N, 2) arrays, original-image px
            areas = [
                cv2.contourArea(p.astype(np.float32)) if len(p) >= 3 else 0.0
                for p in polys
            ]
            # discard non-round blobs (floor/wall) that could win on area alone
            candidates = [
                i for i, p in enumerate(polys)
                if len(p) >= 3 and _circularity(p) >= min_circularity
            ]
            # among what's left, the plate on the bar is the largest by area ->
            # robust against smaller secondary disks in the background
            best = max(candidates, key=lambda i: areas[i]) if candidates else None
            poly = polys[best] if best is not None else None
            if poly is not None and len(poly) >= 3:
                area = areas[best]
                diam = float(2.0 * np.sqrt(area / np.pi)) if area > 0 else np.nan
                m = cv2.moments(poly.astype(np.float32))
                if m["m00"] != 0:
                    cy = float(m["m01"] / m["m00"])
                else:
                    ys = poly[:, 1]
                    cy = float((ys.max() + ys.min()) / 2.0)
        center_y.append(cy)
        diameter.append(diam)

    n = len(center_y)
    t = np.arange(n) / fps
    return TrackResult(fps=fps, t=t, center_y=np.array(center_y), diameter=np.array(diameter))


def normalized_height(center_y: np.ndarray) -> np.ndarray:
    """1 = highest point of the disk center across the video, 0 = lowest."""
    if not np.isfinite(center_y).any():
        return np.full_like(center_y, np.nan)
    y_min, y_max = np.nanmin(center_y), np.nanmax(center_y)
    span = (y_max - y_min) or 1.0
    return (y_max - center_y) / span


def write_track_csv(path: Path, result: TrackResult) -> None:
    norm_h = normalized_height(result.center_y)
    with open(path, "w") as fh:
        fh.write("frame,time_s,center_y_px,height_norm,diameter_px\n")
        for i in range(len(result.t)):
            fh.write(
                f"{i},{result.t[i]:.4f},"
                f"{'' if np.isnan(result.center_y[i]) else f'{result.center_y[i]:.2f}'},"
                f"{'' if np.isnan(norm_h[i]) else f'{norm_h[i]:.4f}'},"
                f"{'' if np.isnan(result.diameter[i]) else f'{result.diameter[i]:.2f}'}\n"
            )
