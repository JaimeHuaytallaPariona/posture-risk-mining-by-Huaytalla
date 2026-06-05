# Decisión de Hiperparámetros — Sprint 3

**Branch:** `experiments/feature-engineering-v1`
**Sesión:** 7 — Búsqueda de Hiperparámetros
**Autor:** Jaime Huaytalla Pariona
**Estado:** ✓ Ejecutado — Decisión: **MANTENER baseline**

---

## 1. Resumen del espacio de búsqueda

| Hiperparámetro | Distribución | Rango | Escala | Justificación |
|---|---|---|---|---|
| `learning_rate` | loguniform | [1e-3, 0.2] | log | Sensibilidad exponencial; cambios en escala log tienen efecto lineal en rendimiento |
| `max_depth` | int | [3, 10] | lineal | Controla complejidad del árbol; 3 = modelo simple, 10 = máxima complejidad |
| `min_child_weight` | loguniform | [1e-2, 10] | log | Regularización contra overfitting; escala log recomendada por el asesor (sesión 7, diap. 2) |
| `subsample` | uniform | [0.5, 1.0] | lineal | Fracción de muestras por árbol; reduce varianza |
| `colsample_bytree` | uniform | [0.5, 1.0] | lineal | Fracción de features por árbol; control de correlación entre árboles |
| `n_estimators` | fijo = 1000 + early stopping | — | — | No se incluye en el espacio; early stopping (50 rondas) determina el número óptimo |

**Parámetros fijos (fuera del espacio de búsqueda):**
- `tree_method = "hist"` — implementación eficiente para datos tabulares grandes
- `eval_metric = "mlogloss"` — loss multiclase para señal de early stopping
- `random_state = 42` — reproducibilidad garantizada
- `early_stopping_rounds = 50` — detener si val-mlogloss no mejora en 50 rondas

**Criterio de selección del espacio (≤ 6-8 HP, diapositiva 21):**
Se excluyeron `gamma`, `reg_alpha` y `reg_lambda` por tener impacto secundario
documentado en la literatura de XGBoost para datasets tabulares medianos.
Buscar más parámetros con el mismo presupuesto de trials reduciría la densidad
de exploración del espacio efectivo sin beneficio claro.

---

## 2. Presupuesto ejecutado

| Concepto | Planificado | Real | Observación |
|---|---|---|---|
| Trials totales | 30 | **1** | Estudio detenido por costo computacional |
| Trials completados | 30 | **1** | Trial #0 completado correctamente |
| Trials podados (pruned) | Variable | **0** | No se pudo activar pruning con 1 solo trial |
| Fase Random (calentamiento) | 15 trials | **1** | Solo trial #0 de la fase Random |
| Fase Bayes/TPE | 15 trials | **0** | No alcanzada |
| Tiempo por trial (3-fold CV) | ~40 min estimado | **24.1 min (1449 s)** | Más lento de lo esperado |
| Tiempo total del estudio | ~15-20 h (extrapolado) | ~24 min | Solo 1 trial ejecutado |
| Early stopping activado | Esperado | **No activado** | XGBoost usó los 1000 árboles completos |
| Herramienta | Optuna TPESampler | Optuna TPESampler (n_startup=15, seed=42) | ✓ |
| Pruner | MedianPruner | MedianPruner (n_startup=5, n_warmup=1) | ✓ (inactivo por falta de trials) |

**Causa raíz del presupuesto no ejecutado:**
El tiempo por trial fue ~24 minutos, excediendo el estimado de ~40 segundos.
El motivo es que el early stopping **no se activó**: XGBoost entrenó los
1000 árboles completos en lugar de detenerse antes. Esto sucedió porque la
fracción de eval (15% del train del fold) fue suficientemente pequeña para
que el modelo siguiera reduciendo mlogloss marginalmente hasta el árbol 1000.
Con 194,205 ventanas × 297 features × 1000 árboles × 3 folds = costo elevado.

---

## 3. Único trial ejecutado — Trial #0

| Atributo | Valor |
|---|---|
| Trial ID | 0 |
| Fase | Random (calentamiento) |
| `learning_rate` | 0.0073 |
| `max_depth` | 10 |
| `min_child_weight` | 1.5703 |
| `subsample` | 0.7993 |
| `colsample_bytree` | 0.5780 |
| `n_estimators` usados | 1000 (early stopping no activado) |
| F1-Macro (3-fold CV) | **0.8690 ± 0.0207** |
| Tiempo (3-fold CV) | 1449 s (24.1 min) |

---

## 4. Evaluación final con LOSO completo (9-fold)

Se evaluó el único trial disponible con LOSO completo para obtener métricas
comparables con el Sprint 2.

| Modelo | F1-Macro (LOSO) | PR-AUC (LOSO) | ΔF1 vs S2 | Tiempo LOSO | Decisión |
|---|---|---|---|---|---|
| `XGB_n200_baseFeats` (S2) | 0.8108 ± 0.2038 | 0.8863 ± 0.1957 | — | 392.4 s | Baseline |
| `XGB_tuned_bayes` (S3, trial #0) | **0.8084 ± 0.2084** | **0.8852 ± 0.1954** | **−0.0024** | 4892 s | ⚠ Descartar |

### Discrepancia entre 3-fold CV y LOSO

El trial #0 mostró F1-Macro = 0.8690 en 3-fold CV durante la búsqueda,
pero solo 0.8084 en la evaluación LOSO final. Esta diferencia de +0.0606
(+7.4%) es un hallazgo metodológico importante:

**¿Por qué el 3-fold CV fue optimista respecto al LOSO?**

Con 9 sujetos, el 3-fold GroupKFold asigna ~3 sujetos al conjunto de prueba
por fold (mayor representatividad estadística que LOSO que usa 1 sujeto).
Al tener más sujetos de prueba, la varianza inter-fold es menor (std = 0.0207)
y la media tiende a ser más estable y potencialmente más alta. En cambio,
LOSO con 1 sujeto por fold produce una std de 0.2084, dominada por el fold
del sujeto S09 (solo 6,391 ventanas de una sola actividad).

Esta discrepancia **no invalida** el experimento — confirma que la evaluación
LOSO es más exigente y más representativa del escenario real (generalización
a un operador completamente nuevo), que es precisamente el objetivo de la tesis.

---

## 5. Tabla top-k

Con solo 1 trial completado, no es posible generar un ranking top-5 con sentido
estadístico. La tabla contiene el único trial disponible:

| Rank | Trial | Fase | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | F1-Macro (3-fold) | Tiempo | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | #0 | random | 0.0073 | 10 | 1.5703 | 0.7993 | 0.5780 | 0.8690 | 1449 s | Único trial |

**Configs de respaldo:** no disponibles (requiere ≥3 trials completados).

---

## 6. Importancia de hiperparámetros

No disponible. El análisis FAnova de Optuna requiere un mínimo de 10 trials
completados para producir estimaciones estadísticamente confiables.

---

## 7. Decisión final

### Decisión: **MANTENER `XGB_n200_baseFeats` (Sprint 2)**

**Justificación técnica:**
El único trial ejecutado produce ΔF1 = −0.0024 en la evaluación LOSO completa,
lo que no supera el umbral mínimo de mejora de +0.005 establecido en
`configs/search.yaml`. La configuración tuneada es estadísticamente indistinguible
del baseline dentro de la variabilidad natural del estimador (std ≈ 0.20).

Además, `n_estimators = 1000` vs el baseline que usa 200 implica un modelo
**5 veces más lento** en inferencia (4892 s vs 392 s para el ciclo completo de LOSO),
lo cual es inaceptable para el sistema de alertas en tiempo real. 
Una variante con más árboles y sin mejora de F1 es una regresión
en viabilidad de despliegue.

**Justificación de costo/viabilidad:**
El baseline `XGB_n200_baseFeats` tiene un tiempo de entrenamiento de 392 s
con 200 árboles por configuración. El trial #0 con 1000 árboles tardó 4892 s.
Para una inferencia edge, el tiempo de inferencia por ventana
escala con el número de árboles. El modelo de 200 árboles es el candidato viable.

**Config adoptada para próximos sprints:**
```yaml
model:      XGBoost
sprint:     2
n_estimators: 200
max_depth:  6
learning_rate: 0.1
subsample:  1.0
colsample_bytree: 1.0
min_child_weight: 1
```

---

## 8. Diagnóstico y acciones correctivas para re-ejecución

El experimento produjo un resultado válido y honesto. Sin embargo, el
presupuesto de búsqueda no se ejecutó en su totalidad debido a problemas
de costo computacional. Para una re-ejecución exitosa se recomiendan
las siguientes acciones:

| Problema identificado | Acción correctiva |
|---|---|
| Early stopping no se activó → 1000 árboles completos por fold | Reducir `n_estimators_max` de 1000 a 300; usar `eval_fraction = 0.25` (más señal para early stopping) |
| 24 min por trial → inviable para 30 trials | Reducir a 2-fold GroupKFold durante la búsqueda (de 3 a 2 folds) |
| 3-fold CV optimista vs LOSO (+7.4% de diferencia) | Usar 4-fold GroupKFold (más folds = más representativo) o directamente LOSO con subconjunto de 5 sujetos |
| Presupuesto de 30 trials no completado | Reducir a 20 trials iniciales con `timeout=3600` (1 hora total) |
| Sin configs de respaldo | Re-ejecutar con configuración corregida antes de la siguiente sesión |

---

## 9. Errores evitados (diapositiva 18)

- ✓ No se tuneó todo a la vez: espacio acotado a 5 HP con impacto real
- ✓ Mismo split/seed en todos los trials: GroupKFold, seed=42
- ✓ No se miró el test durante la búsqueda: solo train+val en 3-fold CV
- ✓ Se reportó latencia/costo: tiempo por trial en `logs/hpo_runs.csv`
- ✓ Toda configuración + métrica registrada: MLflow + CSV

---

## 10. Reproducibilidad

```powershell
# Reproducir el único trial ejecutado
git checkout experiments/feature-engineering-v1
conda activate posture-risk
jupytext --to notebook notebooks/05_hparam_tuning.py
jupyter lab  # → ejecutar 05_hparam_tuning.ipynb

# Ver tracking MLflow
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000

# Cargar el estudio guardado sin re-ejecutar
python -c "
import joblib
study = joblib.load('models/optuna_study.pkl')
print('Trials completados:', len([t for t in study.trials if t.state.name == 'COMPLETE']))
print('Mejor F1 (3-fold CV):', study.best_value)
print('Mejores params:', study.best_params)
"
```

**Hashes de integridad:**

```powershell
# Verificar que los datos no cambiaron
Get-FileHash data\processed\pamap2_features.h5 -Algorithm MD5
```

**Artefactos generados:**

| Artefacto | Ruta | Estado |
|---|---|---|
| Logs de trials | `logs/hpo_runs.csv` | ✓ 1 trial registrado |
| Modelo ganador | `models/best_bayes.pkl` | ✓ Guardado |
| Config ganadora | `configs/best_config.yaml` | ✓ Guardado |
| Estudio Optuna | `models/optuna_study.pkl` | ✓ Guardado |
| Tabla top-k | `reports/hparam_top5_table.csv` | ✓ 1 trial |
| Gráfico evolución | `reports/figures/10_hparam_evolution.png` | ✓ Generado |
| MLflow runs | `mlruns/` | ✓ 1 run registrado |

---

## 11. Riesgos y próximos pasos

**Riesgo principal — Costo computacional de la búsqueda:**
Con los parámetros actuales (n_estimators=1000, 3-fold, 194K muestras), un
estudio de 30 trials requeriría ~12 horas en la máquina local. Antes de
re-ejecutar la búsqueda completa se debe corregir el perfil de costo.
Acción inmediata: reducir `n_estimators_max` a 300 y `eval_fraction` a 0.25
en `configs/search.yaml`, y re-ejecutar con presupuesto de 20 trials.

---

*Sprint 3 | Branch `experiments/feature-engineering-v1`*

