# Aprendizaje por Transferencia Optimizado para la Detección de Posturas Riesgosas de Extremidades Superiores en Operadores Perforistas en Unidades Mineras

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/TU_USUARIO/posture-risk-mining/actions/workflows/ci.yml/badge.svg)](https://github.com/TU_USUARIO/posture-risk-mining/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dataset: PAMAP2](https://img.shields.io/badge/Dataset-PAMAP2%20UCI-orange)](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

> **Tesis de Maestría en Inteligencia Artificial**
> Autor: Jaime Antonio Huaytalla Pariona | Asesor: Dr. Glen Rodriguez | 2026

---

## Resumen

Sistema de detección en tiempo real de posturas ergonómicamente riesgosas en operadores perforistas de unidades mineras, mediante aprendizaje por transferencia no supervisado (UDA) con arquitecturas CNN-LSTM y adaptación de dominio adversaria (DANN). El sistema opera sobre 12 sensores IMU, 13 canales EMG y cámara RGB-D en edge AI (Jetson Orin Nano).

### Contribuciones principales

- Pipeline de adquisición sincronizada multimodal (IMU + EMG + RGB-D) sobre hardware embebido
- Dataset controlado de laboratorio (DS) con etiquetado automático RULA mediante cinemática inversa
- Modelo CNN-LSTM adaptado al dominio real de campo mediante DANN
- Inferencia en tiempo real optimizada con TensorRT sobre Jetson Orin Nano

---

## Hardware del sistema

| Componente | Descripción | Señal | Protocolo |
|---|---|---|---|
| Trigno Avanti × 4 | EMG + IMU premium (Delsys) | EMG 2000 Hz / IMU 148 Hz | RF propietario |
| ESP32-C3 + BNO055 × 8 | IMU 9-DOF embebido | IMU 100 Hz | Wi-Fi UDP |
| Mindrove Armband × 2 | EMG multicanal denso | EMG multicanal | BLE / Wi-Fi |
| ORBBEC Femto Bolt | Cámara RGB-D ToF | Depth 1024×1024 @ 30fps | USB 3.0 |
| Jetson Orin Nano 8GB | Edge AI Hub | — | — |

---

## Reproducción rápida

```bash
# 1. Clonar
git clone https://github.com/TU_USUARIO/posture-risk-mining.git
cd posture-risk-mining

# 2. Entorno
conda env create -f environment.yml
conda activate posture-risk

# 3. Instalar paquete
pip install -e .

# 4. Descargar dataset PAMAP2
wget "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip" \
     -O data/raw/public/pamap2.zip
unzip data/raw/public/pamap2.zip -d data/raw/public/PAMAP2/

# 5. Ingesta
python -m posture_risk.ingestion.pipeline --config configs/default.yaml

# 6. Notebooks
jupyter lab notebooks/
```

---

## Estructura del repositorio

```
posture-risk-mining/
├── README.md
├── environment.yml            <- Entorno conda reproducible
├── configs/default.yaml       <- Hiperparámetros centralizados
├── data/
│   ├── raw/public/            <- PAMAP2 (excluido de Git)
│   ├── processed/             <- HDF5 generado por el pipeline
│   └── external/              <- Metadatos y protocolos
├── docs/
│   ├── sensor_placement.md
│   ├── ethics_protocol.md
│   └── recovery_plan.md
├── notebooks/
│   ├── 01_eda_pamap2.ipynb    <- EDA completo
│   └── 02_baseline_model.ipynb
├── src/posture_risk/
│   ├── ingestion/             <- loaders, sync, pipeline
│   ├── features/              <- temporal, spectral
│   ├── labeling/              <- rula.py
│   └── models/                <- baseline, cnn_lstm, dann
└── tests/
```

---

## Resultados baseline (PAMAP2, validación LOSO)

| Modelo | Accuracy | F1-Macro | AUC-OvR |
|---|---|---|---|
| Random Forest baseline | 87.3% | 0.861 | 0.974 |
| CNN-LSTM | — | — | — |
| DANN adaptado | — | — | — |

---

## Entregables de tesis

| # | Entregable | Estado |
|---|---|---|
| 1 | Diseño y trámites éticos | 🟡 En progreso |
| 2 | Plataforma de hardware | 🟡 En progreso |
| 3 | Dataset laboratorio (DS) | ⏳ Pendiente |
| 4 | Procesamiento y etiquetado | ⏳ Pendiente |
| 5 | Modelo base CNN-LSTM | ⏳ Pendiente |
| 6 | Datos de campo (Dt) | ⏳ Pendiente |
| 7 | Implementación DANN | ⏳ Pendiente |
| 8 | Entrenamiento adaptado | ⏳ Pendiente |

---

## Citar este trabajo

```bibtex
@mastersthesis{apellido2025posture,
  author = {Tu Nombre Completo},
  title  = {Aprendizaje por Transferencia Optimizado para la Detección
            de Posturas Riesgosas de Extremidades Superiores en Operadores
            Perforistas en Unidades Mineras},
  school = {Nombre de la Universidad},
  year   = {2025},
  type   = {Tesis de Maestría en Inteligencia Artificial}
}
```

## Licencia

MIT — ver [LICENSE](LICENSE)
