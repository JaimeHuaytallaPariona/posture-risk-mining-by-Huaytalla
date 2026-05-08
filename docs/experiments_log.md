# Bitácora de Experimentos

**Proyecto:** posture-risk-mining  
**Branch:** `experimentos/baseline-variantes`  
**Fecha de inicio:** 2025

---

## Diseño experimental

**Hipótesis general:** El feature set espectral aporta capacidad discriminativa significativa que justifica su costo computacional.

**Principio metodológico:** A/B con un solo cambio por experimento. Todas las variantes se evalúan bajo idéntica validación LOSO sobre PAMAP2.

---

## Experimentos ejecutados

### EXP-001 — Baseline (referencia)

| Atributo | Valor |
|---|---|
| Modelo | Random Forest (200 árboles, profundidad 20) |
| Feature set | Temporal (162) + Espectral (135) = 297 |
| Ventana | 200 ms con 50% solapamiento |
| Filtrado | Butterworth orden 4, 0.5–40 Hz |
| Validación | LOSO sobre 9 sujetos |
| Class weight | balanced |
| Seed | 42 |

---

### EXP-002 — Variante A: solo features temporales

**Cambio único respecto al baseline:** se eliminan los 5 features espectrales por canal (MDF, MNF, 3 bandas de potencia). El tensor de entrada pasa de 297 a 162 features.

**Pregunta que responde:** ¿el costo computacional de calcular FFT/Welch en el Jetson Orin Nano se justifica con una mejora medible en el rendimiento del modelo?

**Resultado esperado:** caída de F1-Macro entre 0.02 y 0.05. Si la caída es menor a 0.02, podríamos omitir FFT en producción para reducir latencia de inferencia.

---

### EXP-003 — Variante B: clasificador SVM RBF

**Cambio único respecto al baseline:** se reemplaza el Random Forest por un SVM con kernel RBF (`C=1.0`, `gamma=scale`).

**Pregunta que responde:** ¿la no-linealidad explícita del kernel RBF mejora la separabilidad respecto a los árboles ensemble?

**Trade-off conocido:** SVM RBF es significativamente más lento de entrenar y menos interpretable que RF. Solo se justifica si la mejora en F1-Macro es superior a 0.03.

---

### EXP-004 — Variante C: ventana de 400 ms (DOCUMENTADA, NO EJECUTADA)

**Cambio único respecto al baseline:** se duplica el tamaño de ventana de 200 ms a 400 ms.

**Estado:** documentada como trabajo futuro. Requiere re-ejecutar el pipeline de ingesta con `window_ms=400` en `configs/default.yaml`. Se incluye en este documento por completitud metodológica.

**Hipótesis:** ventanas más largas capturan dinámicas más lentas (sostener una postura forzada por más tiempo) pero introducen latencia de detección. El trade-off precisión vs. latencia es un eje de diseño relevante para el sistema embebido en producción.

---

## Confirmación de comparabilidad

Todas las variantes ejecutadas (EXP-001, EXP-002, EXP-003) cumplen las condiciones de comparabilidad estadística:

- ✅ Mismo dataset HDF5 procesado por el mismo pipeline
- ✅ Misma estrategia de validación (LOSO con 9 folds)
- ✅ Misma seed aleatoria (42) para todas las componentes estocásticas
- ✅ Mismo conjunto de sujetos en cada fold
- ✅ Mismas métricas calculadas (F1-Macro, Accuracy, AUC-OvR, AP-Macro)
- ✅ MLflow registra cada experimento con sus parámetros

---

*Bitácora actualizada en cada commit que añada o modifique experimentos.*
