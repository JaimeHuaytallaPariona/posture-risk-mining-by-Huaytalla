# Experiment Branch: `feature-engineering-v1`

> **Sprint:** Proyecto de Investigación 2  
> **Asesor:** Dr. Glen Rodríguez  
> **Autor:** Jaime Huaytalla Pariona  
> **Branch:** `experiments/feature-engineering-v1`  
> **Estado:** En ejecución

---

## 1. Contexto

Tesis: **Aprendizaje por Transferencia Optimizado para la Detección de Posturas Riesgosas de Extremidades Superiores en Operadores Perforistas en Unidades Mineras.**

El baseline establecido en la Semana 4 alcanzó **F1-Macro ≈ 0.86** con validación LOSO sobre PAMAP2 usando Random Forest sobre 297 features estadísticas (temporal + espectral) extraídas de 27 canales IMU.

Este branch ejecuta el experimento A/B exigido por el protocolo del Ing. Glen Rodríguez (sesiones 4 y 5) para evaluar si:

- **Hipótesis 1 (Var1):** añadir features biomecánicas de dominio mejora la métrica central manteniendo el mismo modelo.
- **Hipótesis 2 (Var2):** cambiar el modelo (Random Forest → XGBoost) mejora la métrica central manteniendo las mismas features.

---

## 2. Diseño del experimento

Cumple estrictamente las reglas del asesor:

| Regla | Implementación |
|---|---|
| Un solo cambio por variante | Var1 toca solo features; Var2 toca solo el modelo |
| Mismo split / seed / datos | `GroupKFold(n_splits=9)`, `seed=42`, mismos splits compartidos vía `make_shared_splits()` |
| Máximo 3 variantes | Baseline + Var1 + Var2 |
| Nombrado técnico explícito | `RF_n200_baseFeats`, `RF_n200_biomechFeats`, `XGB_n200_baseFeats` |
| Trazabilidad y logging | `logs/metrics_experimentos.csv` con `exp_id` único por variante |
| Cero leakage | `sklearn.Pipeline` ajusta scaler SOLO en train por fold |

### Tabla de variantes

| Rol | Nombre técnico | Modelo | Features | Cambio único |
|---|---|---|---|---|
| **A (Baseline)** | `RF_n200_baseFeats`     | Random Forest (200 árboles) | 297 estadísticas | — |
| **B (Var1)**     | `RF_n200_biomechFeats`  | Random Forest (200 árboles) | 297 + 73 biomecánicas = 370 | **features** |
| **C (Var2)**     | `XGB_n200_baseFeats`    | XGBoost (200 iteraciones) | 297 estadísticas | **modelo** |

### Features biomecánicas añadidas en Var1 (73 totales)

Por cada uno de los 3 segmentos (muñeca, pecho, tobillo) en PAMAP2:

- **Magnitud del vector aceleración** — intensidad de movimiento (5 stats: mean, std, RMS, max, min)
- **Magnitud del vector giroscopio** — velocidad angular (5 stats)
- **Tilt angle** — ángulo de inclinación respecto a la vertical, **proxy directo de RULA** (5 stats)
- **Jerk magnitude** — derivada de aceleración, suavidad del movimiento (5 stats)
- **Signal Magnitude Area (SMA)** — integral normalizada de magnitud (1 valor)

Total por segmento: **21 features × 3 segmentos = 63 features**

Más features inter-segmento:
- **Ratio hand/chest** — coordinación brazo-tronco (5 stats)
- **Ratio ankle/chest** — coordinación pierna-tronco (5 stats)

Total ratios: **10 features**

**Gran total: 63 + 10 = 73 features biomecánicas**

### Validación

- **Esquema:** GroupKFold por sujeto (equivalente a LOSO con 9 sujetos)
- **Justificación:** evita leakage por sujeto. El mismo operador no puede aparecer en train y test simultáneamente.
- **Seed fijo:** `42` — mismo entre baseline y variantes
- **Métricas reportadas:** media ± desviación estándar entre folds

### Métricas

| Tipo | Métrica | Justificación |
|---|---|---|
| Principal | **F1-Macro** | Resistente al desbalance de clases identificado en el EDA |
| Secundaria | **PR-AUC (OvR)** | Recomendada por el asesor para clasificación con desbalance |
| Secundaria | Accuracy | Métrica clásica de referencia |
| Costo | Tiempo de entrenamiento (s) | Costo computacional total del CV |

---

## 3. Comandos para reproducir

Desde la raíz del proyecto, con el entorno `posture-risk` activado:

```bash
# 1. Verificar que el HDF5 base existe (del baseline original)
ls -la data/processed/pamap2_features.h5

# 2. Generar el HDF5 biomecánico alineado 1:1 con el base
python -m posture_risk.experiments.prepare_biomech_dataset --config configs/default.yaml

# 3. Convertir el notebook a .ipynb
jupytext --to notebook notebooks/03_ab_experiments.py

# 4. Ejecutar el experimento completo
jupyter lab notebooks/03_ab_experiments.ipynb
# (luego: Run → Run All Cells)
```

### Instalación de XGBoost (opcional, recomendado)

Si XGBoost no está disponible, el sistema cae automáticamente a
`HistGradientBoostingClassifier` de sklearn (comparable en rendimiento):

```bash
pip install xgboost==2.1.1
```

---

## 4. Resultados

> **Esta sección se completa tras ejecutar el notebook 03.**
> Los resultados se escriben automáticamente en:
> - `reports/ab_comparison_table.csv` — tabla comparativa
> - `reports/figures/05_ab_pr_curves_comparison.png` — gráfico clave
> - `reports/figures/06_ab_confusion_matrices.png` — matrices de confusión
> - `logs/metrics_experimentos.csv` — historial persistente

### Tabla estándar (rellenar tras ejecutar)

====================================================================================================
TABLA ESTÁNDAR A/B — Resultado del experimento
====================================================================================================
              Modelo        F1-Macro    PR-AUC (OvR)        Accuracy Tiempo (s)  N features                                                                     Notas
   RF_n200_baseFeats 0.7942 ± 0.2003 0.8697 ± 0.1903 0.8325 ± 0.1088      760.6         297                      Baseline reproducido: RF + 297 features estadísticas
RF_n200_biomechFeats 0.7989 ± 0.1948 0.8719 ± 0.1910 0.8427 ± 0.0868      912.2         370 Mismo RF. +73 features biomecánicas (magnitudes, tilt, jerk, SMA, ratios)
  XGB_n200_baseFeats 0.8108 ± 0.2038 0.8863 ± 0.1957 0.8499 ± 0.1074      392.4         297                       Mismas 297 features. Cambio único: modelo → xgboost

### Gráfico clave

![Curvas PR comparadas](../../reports/figures/05_ab_pr_curves_comparison.png)

---

## 5. Decisión final (rellenar tras ejecutar)

Aplicar el criterio del asesor (sesión 5, diapositiva 14):

- **¿Mejora la métrica central?** ΔF1-Macro ≥ 1% absoluto respecto al baseline → adoptar
- **¿El costo es aceptable?** Tiempo de entrenamiento ≤ 3× baseline
- **¿Se mantiene la validez?** Cero leakage confirmado por construcción del pipeline

### Resultado de la decisión

```
[ ] Adoptar Var1 (RF_n200_biomechFeats) — pasar a siguiente sprint con feature set ampliado
[X] Adoptar Var2 (XGB_n200_baseFeats)   — pasar a siguiente sprint con XGBoost
[ ] Adoptar ambas                       — combinar para Var3 en siguiente sprint
[ ] Descartar ambas                     — mantener baseline RF original
```

**Justificación:** (por completar en función de la retroalimentación del asesor)

---

## 6. Confirmación de cero leakage

- **Splits compartidos:** los 9 splits LOSO se generan una sola vez con `make_shared_splits()` y se pasan idénticos a las 3 variantes.
- **StandardScaler dentro del Pipeline:** `sklearn.Pipeline` garantiza que `fit()` solo vea datos de entrenamiento por fold.
- **Features biomecánicas son funciones puras** de la ventana cruda; no dependen de estadísticas del dataset completo.
- **Alineamiento 1:1** verificado por assert al cargar el HDF5 biomecánico: las etiquetas y subject_ids deben coincidir exactamente con el HDF5 base.

---

## 7. Riesgos identificados y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Ratios inter-segmento pueden producir NaN si magnitud del denominador es ≈ 0 | `eps = 1e-8` añadido en cálculo de ratios |
| XGBoost no instalado en entorno del usuario | Fallback automático a `HistGradientBoostingClassifier` con mensaje informativo |
| Ventanas biomecánicas no alineadas con baseline | Assert de alineamiento al cargar; falla rápido si hay desfase |
| Class weight balanced no soportado por XGBoost directamente | Reportar mediante F1-Macro que es insensible al desbalance |

---

## 8. Próximos pasos (siguiente sprint)

> **Tras la decisión final, completar esta sección:**

- **Si se adopta Var1:** investigar interacciones polinómicas entre features biomecánicas y temporales.
- **Si se adopta Var2:** tunear hiperparámetros de XGBoost (max_depth, learning_rate, n_estimators) con búsqueda Bayesiana ligera.
- **Si se descarta:** revisar la pertinencia del feature set actual y considerar reducción dimensional (PCA, selección por permutation importance).
- **Independiente del resultado:** comenzar implementación de CNN-LSTM (Entregable 5 de la tesis) sobre el feature set ganador.

---

## 9. Trazabilidad

| Artefacto | Ubicación |
|---|---|
| Código de experimentos | `src/posture_risk/experiments/` |
| Configuración | `configs/experiments.yaml` |
| Notebook ejecutable | `notebooks/03_ab_experiments.ipynb` |
| Logs persistentes | `logs/metrics_experimentos.csv` |
| Tabla comparativa | `reports/ab_comparison_table.csv` |
| Figuras | `reports/figures/05_ab_*.png` |
| Este documento | `docs/experiments/feature-engineering-v1.md` |

---

*Documento generado en el branch `experiments/feature-engineering-v1`.*
