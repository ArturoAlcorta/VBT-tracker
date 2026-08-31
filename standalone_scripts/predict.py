#!/usr/bin/env python3
"""Ejecuta el modelo entrenado sobre imágenes o un vídeo.

Uso:
    python predict.py ruta/a/imagen.png
    python predict.py ruta/a/video.mp4 --weights ../models/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_DIR / "models" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia con el modelo del disco.")
    parser.add_argument("source", help="Imagen, carpeta o vídeo de entrada.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="Pesos del modelo (por defecto best.pt del run disk-seg-s-v3).")
    parser.add_argument("--conf", type=float, default=0.4, help="Umbral de confianza.")
    parser.add_argument("--device", default="0", help="'0' GPU, 'cpu' CPU.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)
    # la rama one2one (NMS-free) reparte la puntuacion entre anclas vecinas y
    # deja frames por debajo del umbral; one2many + NMS da conf ~0.96 y 1 det/frame
    model.model.model[-1].end2end = False
    model.predict(
        source=args.source,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(PROJECT_DIR / "runs" / "predict"),
        name="disk",
    )
    print(f"\nResultados guardados en: {PROJECT_DIR / 'runs' / 'predict' / 'disk'}")


if __name__ == "__main__":
    main()
