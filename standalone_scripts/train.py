#!/usr/bin/env python3
"""Entrena un modelo YOLO de segmentación para detectar el disco (clase única).

Uso:
    python train.py                 # entrena con los valores por defecto
    python train.py --epochs 200    # sobrescribe algún parámetro
    python train.py --model yolo11s-seg.pt --device cpu

Requiere el paquete `ultralytics` instalado (ver setup.sh / requirements.txt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "disk_dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO-seg para el disco.")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                        help="Modelo base (yolo26n/s/m-seg.pt). Se descarga solo.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Ruta al data.yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8,
                        help="Tamaño de batch. Bájalo si hay OOM de VRAM.")
    parser.add_argument("--device", default="0",
                        help="'0' para la GPU (RTX 3070), 'cpu' para CPU.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Épocas sin mejora antes de parar (early stopping).")
    parser.add_argument("--name", default="disk-seg-s",
                        help="Nombre del run (carpeta en runs/segment/).")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=str(PROJECT_DIR / "runs" / "segment"),
        patience=args.patience,
    )
    print(f"\nEntrenamiento terminado. Pesos en: "
          f"{PROJECT_DIR / 'runs' / 'segment' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
