#!/usr/bin/env python3
"""Sigue el disco en un vídeo con el modelo YOLO-seg entrenado y grafica su
altura vertical normalizada en función del tiempo.

- Altura normalizada: 1 = posición más alta del centro del disco en todo el
  vídeo, 0 = la más baja.
- Anota el diámetro medio del disco (extensión vertical de la máscara, de su
  punto más alto al más bajo) y su desviación típica, como indicador de
  fiabilidad de la detección.

Uso:
    python track_height.py /ruta/al/video.mp4
    python track_height.py video.mp4 --weights ../models/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_DIR / "models" / "best.pt"
DEFAULT_OUTDIR = PROJECT_DIR / "runs" / "tracking"


def circularity(pts: np.ndarray) -> float:
    """4·pi·area / perimetro^2: 1.0 es un circulo perfecto, ~0.2 una mancha."""
    p = pts.astype(np.float32)
    per = cv2.arcLength(p, True)
    return float(4.0 * np.pi * cv2.contourArea(p) / per ** 2) if per else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking de altura del disco.")
    parser.add_argument("source", help="Vídeo de entrada.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Pesos del modelo.")
    parser.add_argument("--conf", type=float, default=0.4, help="Umbral de confianza.")
    parser.add_argument("--min-circ", type=float, default=0.5,
                        help="Circularidad mínima de la máscara para aceptarla (0-1).")
    parser.add_argument("--device", default="0", help="'0' GPU, 'cpu' CPU.")
    parser.add_argument("--batch", type=int, default=8,
                         help="Frames por pasada de inferencia (un source de vídeo con stream=True "
                              "corre a batch=1 salvo que se indique explícitamente).")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Carpeta de salida.")
    parser.add_argument("--save-misses", action="store_true",
                        help="Guarda como imagen los frames donde no se detectó disco.")
    parser.add_argument("--miss-width", type=int, default=1080,
                        help="Ancho máximo (px) de los frames guardados con --save-misses; "
                             "0 = resolución original.")
    return parser.parse_args()


def analyze(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    source = Path(args.source)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = source.stem

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    model = YOLO(args.weights)
    # la rama one2one (NMS-free) reparte la puntuacion entre anclas vecinas y
    # deja frames por debajo del umbral; one2many + NMS da conf ~0.96 y 1 det/frame
    model.model.model[-1].end2end = False
    results = model.track(
        source=str(source), stream=True, conf=args.conf,
        device=args.device, verbose=False, persist=True, batch=args.batch,
    )

    center_y: list[float] = []    # y del centro (px, origen arriba); NaN si no hay
    diameter: list[float] = []    # diámetro equivalente por área (px); NaN si no hay
    diameter_v: list[float] = []  # extensión vertical de la máscara (px); referencia
    frame_h = None

    misses_dir = outdir / f"{stem}_misses"
    if args.save_misses:
        misses_dir.mkdir(parents=True, exist_ok=True)

    for idx, res in enumerate(results):
        if frame_h is None:
            frame_h = res.orig_shape[0]

        cy, diam, diam_v = np.nan, np.nan, np.nan
        if res.masks is not None and len(res.masks) > 0:
            polys = res.masks.xy  # lista de (N, 2) en px de la imagen original
            areas = [cv2.contourArea(p.astype(np.float32)) if len(p) >= 3 else 0.0
                     for p in polys]
            # descartar lo que no sea redondo: manchas del suelo o de la pared
            # salen con área parecida a la del disco y ganarían por área sola
            cand = [i for i, p in enumerate(polys)
                    if len(p) >= 3 and circularity(p) >= args.min_circ]
            # de lo que queda, el disco de la barra es el de mayor área ->
            # robusto frente a discos secundarios del fondo, más pequeños
            best = max(cand, key=lambda i: areas[i]) if cand else None
            poly = polys[best] if best is not None else None
            if poly is not None and len(poly) >= 3:
                ys = poly[:, 1]
                area = areas[best]
                # diámetro equivalente por área (rotación-invariante, estable);
                # menos sensible que la extensión vertical a rebabas/oclusión
                diam = float(2.0 * np.sqrt(area / np.pi)) if area > 0 else np.nan
                diam_v = float(ys.max() - ys.min())  # extensión vertical (referencia)
                m = cv2.moments(poly.astype(np.float32))
                if m["m00"] != 0:
                    cy = float(m["m01"] / m["m00"])
                else:
                    cy = float((ys.max() + ys.min()) / 2.0)
        center_y.append(cy)
        diameter.append(diam)
        diameter_v.append(diam_v)

        if args.save_misses and np.isnan(cy):
            t_s = idx / fps
            img = res.orig_img
            if args.miss_width and img.shape[1] > args.miss_width:
                scale = args.miss_width / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(misses_dir / f"frame_{idx:05d}_t{t_s:05.2f}s.png"), img)

    n = len(center_y)
    t = np.arange(n) / fps
    cy = np.array(center_y)
    diam = np.array(diameter)
    diam_v = np.array(diameter_v)

    detected = ~np.isnan(cy)
    n_det = int(detected.sum())
    if n_det == 0:
        print("El modelo no detectó el disco en ningún frame.")
        return

    # altura normalizada: 1 = más alto (menor y en imagen), 0 = más bajo
    y_min, y_max = np.nanmin(cy), np.nanmax(cy)
    span = (y_max - y_min) or 1.0
    norm_h = (y_max - cy) / span  # NaN se propaga -> gaps en la línea

    diam_mean = float(np.nanmean(diam))
    diam_std = float(np.nanstd(diam))
    det_rate = 100.0 * n_det / n

    # guardar datos crudos
    csv_path = outdir / f"{stem}_altura.csv"
    with open(csv_path, "w") as fh:
        fh.write("frame,tiempo_s,centro_y_px,altura_norm,diametro_area_px,diametro_vertical_px\n")
        for i in range(n):
            fh.write(f"{i},{t[i]:.4f},"
                     f"{'' if np.isnan(cy[i]) else f'{cy[i]:.2f}'},"
                     f"{'' if np.isnan(norm_h[i]) else f'{norm_h[i]:.4f}'},"
                     f"{'' if np.isnan(diam[i]) else f'{diam[i]:.2f}'},"
                     f"{'' if np.isnan(diam_v[i]) else f'{diam_v[i]:.2f}'}\n")

    # gráfica
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, norm_h, "-", color="#c0392b", linewidth=1.6)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura vertical normalizada del centro del disco")
    ax.set_title(f"Trayectoria vertical del disco — {source.name}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    diam_pct = 100.0 * diam_std / diam_mean if diam_mean else float("nan")
    txt = (f"Diámetro medio (por área): {diam_mean:.1f} px  (±{diam_std:.1f} px, {diam_pct:.1f}%)\n"
           f"Detección: {n_det}/{n} frames ({det_rate:.1f}%)\n"
           f"1 = punto más alto · 0 = punto más bajo")
    ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#999"))

    fig.tight_layout()
    png_path = outdir / f"{stem}_altura.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    print("=== Resumen ===")
    print(f"frames totales:      {n}")
    print(f"frames con disco:    {n_det} ({det_rate:.1f}%)")
    print(f"diámetro medio:      {diam_mean:.1f} px")
    print(f"desviación típica:   {diam_std:.1f} px  ({diam_pct:.1f}% del medio)")
    print(f"rango vertical centro: {span:.0f} px")
    print(f"\ngráfica: {png_path}")
    print(f"datos:   {csv_path}")
    if args.save_misses:
        print(f"frames sin detección guardados en: {misses_dir}")


if __name__ == "__main__":
    analyze(parse_args())
