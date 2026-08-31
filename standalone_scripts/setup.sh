#!/bin/bash
# Crea un entorno virtual e instala Ultralytics (YOLO).
# Uso:  ./setup.sh
set -eu

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual en .venv ..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Entorno listo. Para usarlo:"
echo "  source .venv/bin/activate"
echo "  python train.py            # entrenar"
echo "  python predict.py <img>    # inferencia"
