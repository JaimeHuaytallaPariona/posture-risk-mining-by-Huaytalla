# Validación de Zero Leakage y Split Correcto

**Proyecto:** Detección de Posturas Riesgosas en Operadores Perforistas  
**Documento:** Validación metodológica del split de datos  
**Versión:** 1.0

---

## Definición operacional de "Zero Leakage"

Existe data leakage cuando información del conjunto de prueba se filtra (directa o indirectamente) al proceso de entrenamiento, inflando artificialmente las métricas. En el contexto de series de tiempo multimodales con múltiples sujetos, hay tres formas de leakage que debemos prevenir explícitamente.

---

## Tipo 1 — Leakage por sujeto (subject leakage)

**Riesgo:** Si ventanas del mismo sujeto aparecen tanto en train como en test, el modelo memoriza patrones individuales (forma de caminar, calibración del sensor) en lugar de aprender a generalizar.

**Mitigación implementada:**
- Estrategia de validación: **Group K-Fold** con grupos definidos por `subject_id`.
- Implementación específica: **Leave-One-Subject-Out (LOSO)**.
- Verificación programática en cada fold:

```python
train_subjects_in_fold = set(subject_ids[train_mask].tolist())
test_subjects_in_fold  = set(subject_ids[test_mask].tolist())
assert train_subjects_in_fold.isdisjoint(test_subjects_in_fold), \
    f"LEAKAGE DETECTADO en fold {test_sub}"
```

**Resultado:** En cada uno de los 9 folds, los conjuntos de sujetos en train y test son disjuntos. Sujetos del fold 0: train={2,3,4,5,6,7,8,9}, test={1}.

---

## Tipo 2 — Leakage estadístico (preprocessing leakage)

**Riesgo:** Si se calcula la media y desviación estándar para normalización usando todo el dataset (incluyendo test), el modelo recibe información estadística del conjunto de prueba.

**Mitigación implementada:**
- El `StandardScaler` se ajusta (`fit`) **únicamente** con datos de entrenamiento.
- Se aplica (`transform`) al conjunto de prueba sin re-ajustar:

```python
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)   # ajusta media/std solo con train
X_te   = scaler.transform(X_te)        # aplica las estadísticas de train
```

**Por qué importa:** simula exactamente el escenario real de despliegue. Cuando el modelo evalúe a un nuevo operador en la mina, no tendrá acceso a sus estadísticas para normalizar.

---

## Tipo 3 — Leakage temporal (sliding-window leakage)

**Riesgo:** Las ventanas deslizantes con solapamiento del 50% comparten muestras adyacentes. Si dos ventanas consecutivas del mismo sujeto cayeran en train y test, habría correlación temporal directa.

**Mitigación implementada:**
- El split por sujeto (LOSO) es estricto: ningún punto temporal del sujeto de test aparece en train, ni siquiera parcialmente. Como todas las ventanas de un sujeto van al mismo conjunto, no hay posibilidad de solapamiento entre conjuntos.

---

## Confirmación tabular del split

| Fold | Sujeto en test | Ventanas test | Sujetos en train | Disjuntos |
|---|---|---|---|---|
| 0 | 1 | 12,431 | {2,3,4,5,6,7,8,9} | ✅ |
| 1 | 2 | 13,132 | {1,3,4,5,6,7,8,9} | ✅ |
| 2 | 3 | 8,683  | {1,2,4,5,6,7,8,9} | ✅ |
| 3 | 4 | 11,545 | {1,2,3,5,6,7,8,9} | ✅ |
| 4 | 5 | 13,594 | {1,2,3,4,6,7,8,9} | ✅ |
| 5 | 6 | 12,464 | {1,2,3,4,5,7,8,9} | ✅ |
| 6 | 7 | 11,609 | {1,2,3,4,5,6,8,9} | ✅ |
| 7 | 8 | 13,072 | {1,2,3,4,5,6,7,9} | ✅ |
| 8 | 9 | 318    | {1,2,3,4,5,6,7,8} | ✅ |

---

## Justificación de la elección de LOSO

LOSO es la estrategia de validación más exigente disponible para este problema y la más realista. Otras estrategias considerases y descartadas:

**Holdout 80/20 aleatorio** — descartado porque mezcla ventanas del mismo sujeto en train y test, inflando las métricas y no representando el escenario de despliegue real.

**K-Fold estratificado por clase** — descartado por la misma razón: el sujeto puede aparecer en ambos conjuntos.

**StratifiedGroupKFold (k=5)** — alternativa válida pero menos exigente que LOSO. Cada fold tendría aproximadamente 2 sujetos en test. LOSO es preferible porque el costo computacional es manejable (9 folds, ~1-3 min cada uno) y aporta máxima granularidad: una métrica por sujeto en lugar de una por grupo.

---

## Métricas reportadas y por qué

Para evitar que el desbalance distorsione la interpretación, reportamos cuatro métricas complementarias:

**F1-Macro (métrica principal):** promedia el F1 de cada clase con igual peso. Insensible al desbalance porque no pondera por frecuencia. Esta es la métrica que un modelo debe optimizar para detectar correctamente la clase minoritaria de "Riesgo Alto", que es la más importante para el objetivo de la tesis.

**Accuracy (métrica secundaria):** proporción de aciertos. Sensible al desbalance pero útil como sanity check.

**AUC-OvR macro:** área bajo la curva ROC en esquema One-vs-Rest. Mide la capacidad discriminativa independiente del umbral.

**Average Precision (AP) macro:** área bajo la curva Precision-Recall. Es la métrica más robusta bajo desbalance severo y la que se reporta como gráfico central.

---

*Documento auditable. Última actualización: 2025.*
