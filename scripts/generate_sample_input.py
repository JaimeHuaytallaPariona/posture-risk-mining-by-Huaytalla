"""
scripts/generate_sample_input.py
----------------------------------
Genera el archivo data/examples/sample_input.json con features REALES
tomadas del sample_windows.npz que produce train_and_persist.py.

Selecciona 5 ventanas del dataset con distribución de clases balanceada:
  - 2 de riesgo bajo (clase 0)
  - 2 de riesgo medio (clase 1)
  - 1 de riesgo alto (clase 2)

Uso:
    python scripts/generate_sample_input.py
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_NPZ   = PROJECT_ROOT / "data" / "examples" / "sample_windows.npz"
SAMPLE_JSON  = PROJECT_ROOT / "data" / "examples" / "sample_input.json"


def main():
    if not SAMPLE_NPZ.exists():
        raise FileNotFoundError(
            f"No se encuentra {SAMPLE_NPZ}. "
            "Ejecute primero: python -m scripts.train_and_persist"
        )

    print(f"Cargando {SAMPLE_NPZ}...")
    data = np.load(SAMPLE_NPZ)
    X = data["X"]
    y = data["y"]
    print(f"  Shape: X={X.shape}, y={y.shape}")

    # Seleccionar 2 bajo + 2 medio + 1 alto (deterministamente, seed=42)
    rng = np.random.RandomState(42)
    selected = []
    for cls, n in [(0, 2), (1, 2), (2, 1)]:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) == 0:
            print(f"  Advertencia: no hay ventanas con clase {cls}")
            continue
        chosen = rng.choice(cls_indices, size=min(n, len(cls_indices)), replace=False)
        for idx in chosen:
            selected.append((int(idx), int(cls)))

    print(f"  Seleccionadas {len(selected)} ventanas")

    # Construir estructura BatchPredictionInput
    class_names = {0: "baja", 1: "media", 2: "alta"}
    now = datetime.now(timezone.utc)
    windows = []
    for i, (idx, cls) in enumerate(selected):
        windows.append({
            "request_id":   f"sample-{i+1:03d}-{class_names[cls]}",
            "features":     [float(v) for v in X[idx].tolist()],
            "subject_hint": 100 + i + 1,
            "timestamp":    now.replace(microsecond=i * 200000).isoformat(),
        })

    batch = {
        "batch_id": f"sample-batch-{uuid.uuid4().hex[:12]}",
        "windows":  windows,
    }

    with open(SAMPLE_JSON, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"\n✓ sample_input.json generado: {SAMPLE_JSON}")
    print(f"  Ventanas: {len(windows)}")
    print(f"  Tamaño: {SAMPLE_JSON.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
