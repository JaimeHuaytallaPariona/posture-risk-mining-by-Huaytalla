# Sprint 3 — Búsqueda de Hiperparámetros para XGBoost

> **Sesión 7:** Búsqueda de hiperparámetros (Ing. Glen Rodríguez)
> **Branch:** `experiments/feature-engineering-v1`
> **Modelo a optimizar:** `XGB_n200_baseFeats` (F1-Macro = 0.8108)

---

## Cumplimiento del entregable

| Item del asesor | Implementación |
|---|---|
| Experimentos comparables (Random/Bayes) con pruning/early stopping | 2 estudios Optuna: RandomSampler + TPESampler con warm-up. MedianPruner + XGBoost `early_stopping_rounds=20`. |
| Logs + artefactos guardados | `logs/hpo_runs.csv` con campos del slide 16; estudios SQLite persistentes; modelo ganador en JSON. |
| Tabla top-k y gráfico de evolución | Tabla top-10 + gráfico best-so-far comparando estudios + gráficos de importancia (fANOVA). |
| Resumen del espacio, presupuesto y decisión | Sección final del notebook con espacio definido, trials gastados, config ganadora + 2 respaldos, decisión adoptar/descartar. |

---

## Setup y ejecución

### 1. Verificar Optuna instalado

```powershell
conda activate posture-risk
python -c "import optuna; print(optuna.__version__)"
```

Debe mostrar `3.6.1` o superior. Si no:
```powershell
pip install optuna==3.6.1
```

### 2. Descomprimir el ZIP sobre la raíz del proyecto

Los archivos nuevos son:
```
configs/hpo_search_space.yaml                       ← NUEVO
src/posture_risk/experiments/hpo.py                 ← NUEVO
src/posture_risk/experiments/hpo_reporting.py       ← NUEVO
src/posture_risk/experiments/__init__.py            ← ACTUALIZADO (expone HPO)
notebooks/05_hpo_xgboost.py                         ← NUEVO
docs/experiments/sprint3-hpo.md                     ← NUEVO (este archivo)
```

### 3. Convertir notebook y ejecutar

```powershell
jupytext --to notebook notebooks/05_hpo_xgboost.py
jupyter lab
```

Abre `notebooks/05_hpo_xgboost.ipynb`, verifica el kernel **Python (posture-risk)** y ejecuta `Run → Run All Cells`.

**Tiempo estimado: ~90 minutos** (60 trials × 3 folds × ~30s/fold).

> **Puedes interrumpir y reanudar.** Optuna persiste los trials en SQLite (`optuna_studies/*.db`). Si cierras JupyterLab a mitad del experimento y lo vuelves a ejecutar, retoma desde donde quedó.

### 4. Commit al branch

```powershell
git add .
git commit -m "feat(sprint3): HPO XGBoost - Random vs Bayesian con Optuna"
git push
```

---

## Espacio de búsqueda (slide 10 de la sesión 7)

| Hiperparámetro | Rango | Escala | Justificación |
|---|---|---|---|
| `learning_rate` | [1e-3, 0.2] | log | Slide 5: log-scale obligatorio para `lr` |
| `max_depth` | [3, 10] | int | Profundidad típica para XGBoost |
| `min_child_weight` | [1e-2, 10] | log | Slide 5: log-scale para regularización |
| `subsample` | [0.5, 1.0] | uniform | Fracción de muestras por árbol |
| `colsample_bytree` | [0.5, 1.0] | uniform | Fracción de features por árbol |

**Hiperparámetro NO optimizado:** `n_estimators=500` fijo con `early_stopping_rounds=20`. Esto sigue la recomendación explícita de la slide 10 y 15 del asesor.

---

## Presupuesto

| Concepto | Valor |
|---|---|
| Trials Random | 30 |
| Trials Bayesian (TPE) | 30 (10 warm-up random + 20 TPE) |
| **Total trials** | **60** |
| CV interna (tuning) | GroupKFold(3) por sujeto |
| CV final (config ganadora) | GroupKFold(9) por sujeto (LOSO) |
| Pruning | MedianPruner (n_startup=5, n_warmup=1) |
| Early stopping (XGBoost) | `early_stopping_rounds=20` |
| Seed | 42 (fijo en los 2 estudios) |

---

## Artefactos generados

| Archivo | Descripción |
|---|---|
| `logs/hpo_runs.csv` | 60 filas con trial_id, params, metric, time, state |
| `optuna_studies/random_study.db` | Estudio Random persistente (SQLite) |
| `optuna_studies/bayesian_study.db` | Estudio Bayesian persistente (SQLite) |
| `models/best_xgb_config.json` | Configuración ganadora en JSON |
| `reports/hpo_results_top10.csv` | Tabla top-10 |
| `reports/figures/10_hpo_best_so_far.png` | Gráfico de evolución |
| `reports/figures/11_hpo_importances_random.png` | Importancia fANOVA (Random) |
| `reports/figures/12_hpo_importances_bayesian.png` | Importancia fANOVA (Bayesian) |

---

## Decisión adoptar / descartar

El criterio formal está en `configs/hpo_search_space.yaml`:

```yaml
decision:
  min_improvement: 0.005    # +0.5% absoluto sobre baseline
  n_backup_configs: 2
```

**Adoptar** si la config ganadora (validada con LOSO 9-fold) mejora el F1-Macro del baseline en ≥ 0.005 puntos absolutos.

**Descartar** si la mejora es menor: significa que el baseline ya está cerca del óptimo del espacio actual, y el siguiente paso debe ser cambiar de arquitectura (CNN-LSTM) o ampliar el dataset.

El notebook imprime la decisión final automáticamente con justificación técnica.

---

## Notas técnicas para el asesor

1. **Cero leakage:** el StandardScaler dentro del pipeline ajusta sus parámetros solo en los datos de entrenamiento de cada fold. El early stopping de XGBoost usa un subconjunto adicional (20% de grupos del train) sin tocar el fold de validación externa.

2. **Mismo split entre trials:** los splits de GroupKFold se generan con `seed=42` y son idénticos para todos los trials de ambos estudios. Esto garantiza que la diferencia entre trials sea atribuible solo a los hiperparámetros.

3. **Pruning conservador:** `MedianPruner` con 5 trials de calentamiento. Esto evita podar trials prometedoras antes de tener suficiente evidencia.

4. **Comparación justa entre estudios:** los dos estudios se ejecutan con el mismo seed, mismo objective, mismos splits internos. La diferencia es exclusivamente el sampler.

5. **Persistencia:** los estudios SQLite permiten reanudar después de interrupciones sin perder progreso. Esto es importante para experimentos largos en máquinas locales.
