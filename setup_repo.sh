#!/bin/bash
# =============================================================
# setup_repo.sh — Inicializa el repositorio GitHub desde cero
# Ejecutar UNA SOLA VEZ después de clonar el repositorio.
# =============================================================
set -e

echo "=================================================="
echo "  SETUP — posture-risk-mining"
echo "=================================================="

# 1. Inicializar git si no existe
if [ ! -d ".git" ]; then
    git init
    echo "[OK] Git inicializado"
fi

# 2. Entorno conda
echo "Creando entorno conda..."
conda env create -f environment.yml
echo "[OK] Entorno 'posture-risk' creado"

# 3. Activar e instalar paquete
echo "Para continuar, activa el entorno y ejecuta:"
echo ""
echo "  conda activate posture-risk"
echo "  pip install -e ."
echo ""
echo "Luego descarga el dataset:"
echo "  wget 'https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip' \\"
echo "       -O data/raw/public/pamap2.zip"
echo "  unzip data/raw/public/pamap2.zip -d data/raw/public/PAMAP2/"
echo ""
echo "Y ejecuta el pipeline:"
echo "  python -m posture_risk.ingestion.pipeline"
echo ""
echo "[OK] Setup completado"
