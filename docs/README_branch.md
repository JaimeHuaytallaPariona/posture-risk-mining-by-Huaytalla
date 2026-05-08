# Branch: `experimentos/baseline-variantes`

> **Propósito:** evaluación A/B del baseline contra variantes controladas para identificar las decisiones de diseño que mejor balancean rendimiento, costo computacional e interpretabilidad.

---

## Contenido del branch

Este branch añade al proyecto principal:

```
src/posture_risk/experiments/
├── __init__.py
├── runner.py              ← motor de experimentos A/B con MLflow
└── compare.py             ← genera tabla comparativa + curva PR

notebooks/
└── 03_experimentos_ab.ipynb

docs/
├── zero_leakage_validation.md   ← validación metodológica del split
├── experiments_log.md           ← bitácora con diseño y resultados
└── README_branch.md             ← este archivo

reports/
├── experiments/                  ← outputs de cada variante (npz, json)
└── figures/
    └── experiments_comparison.png  ← gráfico clave del entregable
```

---

## Reproducción

```bash
# 1. Activar entorno
conda activate posture-risk

# 2. Asegurarse de tener el dataset procesado de la rama main
ls data/processed/pamap2_features.h5

# 3. Ejecutar todas las variantes
python -m posture_risk.experiments.runner --variant all

# 4. Generar tabla comparativa y figura
python -m posture_risk.experiments.compare

# 5. (Opcional) Ver experimentos en MLflow UI
mlflow ui
# Abrir http://localhost:5000 en navegador
```

---

## Resumen de entregables

| Entregable | Archivo |
|---|---|
| Variantes A/B documentadas | `docs/experiments_log.md` |
| Tabla comparativa estándar | `reports/experiments/comparison_table.md` |
| Gráfico clave (curva PR) | `reports/figures/experiments_comparison.png` |
| Validación zero leakage | `docs/zero_leakage_validation.md` |
| Tracking de runs | MLflow UI (carpeta `mlruns/`) |

---

## Hallazgos esperados

1. **F1-Macro como métrica principal:** el desbalance de clases en PAMAP2 hace que Accuracy sea engañosa. F1-Macro es la métrica que decide qué variante es mejor.

2. **Var A (sin features espectrales):** anticipamos una caída moderada (~0.02–0.05). Si la caída es marginal, podemos descartar FFT en producción para reducir latencia en el Jetson.

3. **Var B (SVM-RBF):** anticipamos rendimiento similar al RF pero con tiempos de entrenamiento 3–5× mayores. El RF se mantendrá como elección por defecto salvo evidencia clara en contra.

---

## Próximos pasos (post-merge)

Una vez validado este branch, los siguientes pasos planificados son:

- Implementar arquitectura CNN-LSTM (Entregable 5 de la tesis)
- Re-ejecutar comparación A/B incluyendo CNN-LSTM como nueva variante
- Iniciar implementación de DANN (Entregable 7) sobre la mejor variante baseline

---

*Branch listo para Pull Request hacia `main` cuando se completen los 3 experimentos.*
