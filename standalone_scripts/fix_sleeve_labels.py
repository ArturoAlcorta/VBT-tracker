#!/usr/bin/env python3
"""Rellena la muesca que SAM recorto donde el manguito de la barra sobresale
del disco, sustituyendo el poligono por su casco convexo.

La mascara debe significar siempre lo mismo: la silueta circular del disco,
con el manguito/cierre incluido cuando cae dentro de ella. En 4 etiquetas SAM
siguio el contorno del manguito y le comio area al disco (solidez ~0.91),
lo que sesga el diametro equivalente por area un 5% a la baja.

Uso:
    python fix_sleeve_labels.py            # muestra que haria
    python fix_sleeve_labels.py --apply    # lo escribe
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(__file__).resolve().parent.parent / "disk_dataset"
SOLIDITY_MIN = 0.97   # por debajo -> el poligono tiene una muesca
CIRC_HULL_MIN = 0.95  # el casco debe quedar practicamente circular


def polygon_stats(pts: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Devuelve (solidez, circularidad del casco, casco convexo)."""
    hull = cv2.convexHull(pts)
    area, area_hull = cv2.contourArea(pts), cv2.contourArea(hull)
    solidity = area / area_hull if area_hull else 0.0
    per_hull = cv2.arcLength(hull, True)
    circ_hull = 4 * np.pi * area_hull / per_hull ** 2 if per_hull else 0.0
    return solidity, circ_hull, hull


def main(apply: bool) -> None:
    fixed = skipped = 0
    for split in ("train", "val"):
        for label_path in sorted(glob.glob(str(DATASET / "labels" / split / "*.txt"))):
            stem = Path(label_path).stem
            images = glob.glob(str(DATASET / "images" / split / f"{stem}.*"))
            if not images:
                continue
            h, w = cv2.imread(images[0]).shape[:2]

            out_lines, changed = [], False
            for line in open(label_path):
                parts = line.split()
                if len(parts) < 7:
                    out_lines.append(line.rstrip("\n"))
                    continue
                norm = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                pts = (norm * [w, h]).astype(np.float32)
                solidity, circ_hull, hull = polygon_stats(pts)

                if solidity >= SOLIDITY_MIN:
                    out_lines.append(line.rstrip("\n"))
                    continue
                if circ_hull < CIRC_HULL_MIN:
                    # el casco no es un circulo -> la mascara esta mal de origen,
                    # no es una muesca del manguito; no se toca
                    print(f"  OMITIDO {stem}: casco no circular ({circ_hull:.3f}), revisar a mano")
                    out_lines.append(line.rstrip("\n"))
                    skipped += 1
                    continue

                hull_norm = (hull.reshape(-1, 2) / [w, h]).clip(0.0, 1.0)
                coords = " ".join(f"{v:.6f}" for v in hull_norm.flatten())
                out_lines.append(f"{parts[0]} {coords}")
                changed = True
                fixed += 1
                d = 2 * np.sqrt(cv2.contourArea(pts) / np.pi)
                dh = 2 * np.sqrt(cv2.contourArea(hull) / np.pi)
                print(f"  {stem}: solidez {solidity:.3f} -> 1.000, "
                      f"circularidad casco {circ_hull:.3f}, diametro {d:.1f} -> {dh:.1f} px "
                      f"(+{100*(dh-d)/d:.1f}%), {len(pts)} -> {len(hull)} puntos")

            if changed and apply:
                Path(label_path).write_text("\n".join(out_lines) + "\n")

    print(f"\n{fixed} etiquetas {'corregidas' if apply else 'a corregir'}, {skipped} omitidas.")
    if not apply:
        print("Ejecuta con --apply para escribir los cambios.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Escribe los cambios.")
    main(ap.parse_args().apply)
