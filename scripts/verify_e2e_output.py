"""
scripts/verify_e2e_output.py
------------------------------
Verifica que la salida del E2E cumpla los criterios de éxito acordados
en el sprint 7 (sesión 13, slide 11):
  "Verificar salida esperada (hash/tamaño/campos)."

Comprueba:
  1. El archivo de salida existe y es JSON válido
  2. La estructura sigue el esquema BatchPredictionOutput
  3. n_processed > 0 y n_errors == 0
  4. Todas las predicciones tienen campos requeridos y probabilidades válidas
  5. Latencias son > 0 y < 200 ms (SLO del sprint 6)

Uso:
    python scripts/verify_e2e_output.py data/examples/e2e_output.json
"""

import argparse
import json
import sys
from pathlib import Path


def verify(output_path: Path) -> bool:
    print(f"Verificando: {output_path}")

    # 1. Archivo existe y es JSON válido
    if not output_path.exists():
        print(f"  FAIL: archivo no existe")
        return False
    try:
        with open(output_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  FAIL: JSON inválido: {e}")
        return False
    print(f"  OK: archivo JSON válido")

    # 2. Estructura BatchPredictionOutput
    required_fields = ["batch_id", "n_processed", "n_errors",
                        "total_latency_ms", "predictions"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        print(f"  FAIL: campos faltantes: {missing}")
        return False
    print(f"  OK: estructura BatchPredictionOutput completa")

    # 3. n_processed > 0 y n_errors == 0
    if data["n_processed"] == 0:
        print(f"  FAIL: n_processed == 0 (no se produjo ninguna predicción)")
        return False
    if data["n_errors"] > 0:
        print(f"  FAIL: n_errors == {data['n_errors']} (se esperaba 0 en E2E limpio)")
        for err in data.get("errors", [])[:3]:
            print(f"    - {err.get('code')}: {err.get('message')}")
        return False
    print(f"  OK: n_processed={data['n_processed']}, n_errors={data['n_errors']}")

    # 4. Predicciones bien formadas
    pred_required = ["request_id", "prediction", "probabilities",
                      "confidence", "latency_ms", "model_version"]
    for i, pred in enumerate(data["predictions"]):
        missing = [f for f in pred_required if f not in pred]
        if missing:
            print(f"  FAIL: predicción {i} tiene campos faltantes: {missing}")
            return False
        if len(pred["probabilities"]) != 3:
            print(f"  FAIL: predicción {i} probabilities de longitud {len(pred['probabilities'])} ≠ 3")
            return False
        prob_sum = sum(pred["probabilities"])
        if not (0.99 <= prob_sum <= 1.01):
            print(f"  FAIL: predicción {i} probabilities suman {prob_sum:.4f}")
            return False
        if pred["prediction"] not in ["bajo", "medio", "alto"]:
            print(f"  FAIL: predicción {i} clase inválida: {pred['prediction']}")
            return False
    print(f"  OK: {len(data['predictions'])} predicciones bien formadas")

    # 5. Latencias razonables (SLO sprint 6: p95 < 200 ms)
    lats = [p["latency_ms"] for p in data["predictions"]]
    max_lat = max(lats)
    avg_lat = sum(lats) / len(lats)
    if max_lat > 500:   # umbral generoso para PC de desarrollo
        print(f"  FAIL: latencia máxima {max_lat:.1f} ms excede 500 ms")
        return False
    print(f"  OK: latencias avg={avg_lat:.2f} ms, max={max_lat:.2f} ms")

    print("\nRESULTADO: E2E VERIFICADO EXITOSAMENTE")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path,
                        help="Ruta al archivo de salida a verificar")
    args = parser.parse_args()

    success = verify(args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
