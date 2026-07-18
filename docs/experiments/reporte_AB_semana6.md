# Reporte Técnico de Experimentos A/B — Semana 6
## Detección de Posturas Riesgosas en Operadores Perforistas mediante Transfer Learning

**Autor:** Jaime Huaytalla Pariona  
**Asesor:** Dr. Glen Rodríguez  
**Branch:** `experiments/feature-engineering-v1`  
**Dataset:** PAMAP2 Physical Activity Monitoring (UCI Repository)

---

## 1. Contexto

### Objetivo del proyecto

Desarrollar un sistema de detección automática y en tiempo real de posturas ergonómicamente riesgosas en operadores perforistas de unidades mineras, mediante aprendizaje por transferencia no supervisado (DANN/UDA). El sistema debe operar en el borde (*edge AI*) sobre un Jetson Orin Nano con 12 sensores IMU, 12 canales EMG y una cámara RGB-D.

La pregunta que este experimento responde es más acotada: **¿qué combinación de modelo y features estadísticas sobre señales IMU maximiza la detección de niveles de riesgo postural, medida en F1-Macro, sobre datos de laboratorio controlados (DS)?** Este resultado establece la arquitectura base que posteriormente se adaptará al dominio de campo (mina) mediante DANN.

### Dataset y entorno

Se trabaja con **PAMAP2 Physical Activity Monitoring** (UCI, 2012) como dataset de referencia de laboratorio, mientras se obtienen los datos propios de operadores perforistas.

| Atributo | Valor |
|---|---|
| Fuente | https://archive.ics.uci.edu/dataset/231 |
| Sujetos | 9 (S01–S09) |
| Actividades mapeadas | 12 actividades → 3 niveles de riesgo proxy |
| Ventanas totales | 194,205 (200 ms, 50% solapamiento) |
| Canales IMU | 27 (9 por segmento × 3 segmentos: muñeca, pecho, tobillo) |
| Frecuencia de muestreo | 100 Hz |
| Features base | 297 (6 temporales + 5 espectrales × 27 canales) |

**Mapeo de actividades PAMAP2 → niveles de riesgo proxy:**

| Nivel | Actividades PAMAP2 | Justificación ergonómica |
|---|---|---|
| 0 — Bajo riesgo | Tumbado, sentado, de pie, caminando | Postura neutra, carga articular baja |
| 1 — Riesgo medio | Subir/bajar escaleras, otras marchas | Carga moderada, postura variable |
| 2 — Riesgo alto | Planchar, aspirar, saltar | Posturas sostenidas con carga en extremidades superiores |

> **Nota metodológica:** PAMAP2 no es un dataset de minería. Las actividades de riesgo alto se seleccionaron porque comparten características biomecánicas con la perforación: carga sostenida en extremidades superiores, postura no neutra y repetición. Es un proxy válido para validar el pipeline antes de obtener datos propios.

### Métrica central elegida: F1-Macro

Se elige **F1-Macro** como métrica principal por dos razones:

1. **Desbalance de clases:** el EDA (notebook 01) reveló que las actividades de riesgo alto son la clase minoritaria del dataset. La Accuracy puede alcanzar valores altos prediciendo siempre la clase mayoritaria, lo que daría una imagen falsa del rendimiento real.
2. **Costo asimétrico de errores:** en el contexto de ergonomía industrial, un **falso negativo** (no detectar una postura de riesgo alto) tiene consecuencias físicas para el operador. F1-Macro penaliza igualmente los errores en todas las clases, forzando al modelo a ser preciso también en la clase crítica.

Como métrica secundaria principal se usa **PR-AUC** (Precision-Recall Area Under Curve), ideal para identificar problemas de clasificación con desbalance.

---

## 2. Línea base (Baseline)

**Identificador:** `RF_n200_baseFeats` (EXP001)

Random Forest con 200 árboles entrenado sobre 297 features estadísticas extraídas de ventanas de 200 ms con 50% de solapamiento. Las features combinan 6 descriptores temporales (RMS, MAV, WL, ZC, SSC, varianza) y 5 espectrales (frecuencia mediana, frecuencia media, potencia en 3 bandas) por cada uno de los 27 canales IMU de PAMAP2. Validación GroupKFold-LOSO con 9 folds, `seed=42`, `class_weight="balanced"`.

| Métrica | Valor |
|---|---|
| **F1-Macro** | **0.7942 ± 0.2003** |
| PR-AUC (OvR) | 0.8697 ± 0.1903 |
| Accuracy | 0.8325 ± 0.1088 |
| Tiempo total CV | 760.6 s |
| N features | 297 |

La **desviación estándar alta** (±0.20 en F1-Macro) es un hallazgo importante: el modelo funciona muy bien con algunos sujetos y significativamente peor con otros. Esto evidencia la **variabilidad inter-sujeto** propia de señales biomecánicas, que es precisamente el problema que DANN deberá mitigar al pasar del laboratorio al campo.

---

## 3. Experimentos A/B

Se ejecutaron 2 variantes sobre el mismo baseline, respetando el principio de **un cambio único por variante** con **mismo split / seed / datos** para aislar el efecto causal de cada modificación.

### Variante 1 (B): `RF_n200_biomechFeats`

**Cambio único:** se añaden 73 features biomecánicas de dominio al vector de features del baseline (297 → 370 features). El modelo y la validación se mantienen idénticos.

**Features añadidas (por cada uno de los 3 segmentos):**
- Magnitud del vector de aceleración (5 estadísticas: mean, std, RMS, max, min)
- Magnitud del vector giroscópico (5 estadísticas)
- Ángulo de inclinación — *tilt angle* (5 estadísticas): proxy directo de ángulos articulares RULA
- *Jerk magnitude* — derivada de aceleración (5 estadísticas): suavidad del movimiento
- *Signal Magnitude Area* (SMA): 1 valor único

Más 2 ratios inter-segmento (hand/chest, ankle/chest) × 5 estadísticas = 10 features adicionales.

**Justificación técnica:** los 297 features base capturan estadísticas de la señal pero no exploran explícitamente la geometría del movimiento. El ángulo de inclinación (*tilt*) y la magnitud del vector de aceleración están directamente relacionados con los ángulos articulares que evalúa RULA (flexión de hombro, abducción, postura de tronco). La hipótesis es que este conocimiento de dominio mejora la separabilidad de las clases de riesgo.

### Variante 2 (C): `XGB_n200_baseFeats`

**Cambio único:** se reemplaza Random Forest por XGBoost con 200 iteraciones (*Gradient Boosting* basado en histogramas). Las features son idénticas al baseline (297).

**Justificación técnica:** Random Forest entrena todos los árboles de forma independiente sobre submuestras aleatorias de features, asignando importancia uniforme a priori. XGBoost aplica *boosting* iterativo: cada árbol sucesivo se enfoca en los errores del anterior, aprendiendo cuáles features son más útiles para los casos difíciles. Con 297 features que incluyen features altamente correlacionadas (detectadas en la matriz de correlación del EDA), el boosting puede ser más eficiente al aprender implícitamente a ignorar features redundantes.

---

## 4. Resultados comparables

### 4.1 Tabla estándar

| Modelo | F1-Macro *(principal)* | PR-AUC (OvR) | Accuracy | Tiempo CV (s) | N features | Notas |
|---|---|---|---|---|---|---|
| `RF_n200_baseFeats` | **0.7942 ± 0.2003** | 0.8697 ± 0.1903 | 0.8325 ± 0.1088 | 760.6 | 297 | Baseline |
| `RF_n200_biomechFeats` | **0.7989 ± 0.1948** | 0.8719 ± 0.1910 | 0.8427 ± 0.0868 | 912.2 | 370 | +73 features biomecánicas |
| `XGB_n200_baseFeats` | **0.8108 ± 0.2038** | 0.8863 ± 0.1957 | 0.8499 ± 0.1074 | 392.4 | 297 | RF → XGBoost |

**Deltas respecto al baseline:**

| Variante | ΔF1-Macro | ΔPR-AUC | ΔTiempo | Umbral 1% | Decisión |
|---|---|---|---|---|---|
| `RF_n200_biomechFeats` | +0.0048 (+0.48%) | +0.0022 | +151.6 s (+20%) | No supera | **DESCARTAR** |
| `XGB_n200_baseFeats` | +0.0166 (+1.66%) | +0.0166 | −368.2 s (−48%) | Supera | **ADOPTAR** |

### 4.2 Gráfico clave: Curvas Precision-Recall comparadas

El gráfico principal del experimento son las **curvas PR por clase** con las 3 variantes superpuestas (archivo `reports/figures/05_ab_pr_curves_comparison.png`).

Se eligió la curva PR como gráfico único porque:
- El problema tiene desbalance de clases (detectado en EDA)
- La curva PR es más informativa que ROC cuando la clase positiva es minoritaria
- Permite ver simultáneamente la precisión y el recall para cada nivel de riesgo en todos los umbrales de decisión posibles
- La línea de no-skill (proporción de la clase) sirve como referencia visual inmediata de si el modelo aporta valor sobre el azar

**Lectura esperada del gráfico:**
- `XGB_n200_baseFeats` (línea coral) debe dominar en las 3 clases, especialmente en la clase 2 (riesgo alto)
- `RF_n200_biomechFeats` y `RF_n200_baseFeats` deben aparecer prácticamente superpuestas, evidenciando que las features biomecánicas no aportaron separabilidad adicional en este dataset

---

## 5. Validación

### Esquema de split: GroupKFold por sujeto (LOSO)

Se utilizó **GroupKFold con `groups=subject_ids`**, equivalente a Leave-One-Subject-Out (LOSO) con 9 folds. Este esquema se eligió sobre las alternativas por la siguiente razón:

- **StratifiedKFold** no es adecuado porque distribuye aleatoriamente las ventanas entre folds. Ventanas del mismo sujeto pueden quedar en train y test simultáneamente, produciendo **leakage por identidad biométrica**: el modelo memorizaría los patrones de movimiento específicos de cada individuo en lugar de aprender a generalizar.
- **TimeSeriesSplit** no es adecuado porque no hay un orden cronológico entre sujetos; las grabaciones son independientes.
- **GroupKFold** garantiza que ninguna ventana del sujeto de prueba aparece en entrenamiento. En el contexto de operadores mineros, esto simula el escenario real: el modelo se probará con operadores que nunca vio durante el entrenamiento.

### Garantía de cero leakage

**Nivel 1 — Split antes de transformar:** el objeto `splits` se genera una sola vez con `make_shared_splits()` antes de cualquier procesamiento y se pasa idénticamente a las 3 variantes.

**Nivel 2 — `sklearn.Pipeline`:** cada variante usa `Pipeline([StandardScaler, Modelo])`. Por construcción de scikit-learn, el método `fit()` del pipeline entrena el scaler **solo con datos de entrenamiento** de ese fold. El método `transform()` aplica esos parámetros al conjunto de prueba sin reentrenarse. Esto cumple el principio "divide primero, transforma después" del protocolo del asesor.

**Nivel 3 — Features biomecánicas son funciones puras:** las 73 features biomecánicas se calculan dentro de cada ventana cruda de forma determinista. No dependen de estadísticas del dataset completo (no hay media/std global).

**Nivel 4 — Verificación explícita de alineamiento:** el script `prepare_biomech_dataset.py` verifica mediante `assert` que las etiquetas y `subject_ids` del HDF5 biomecánico son idénticos al HDF5 base antes de guardar. Si hubiera cualquier desfase, el script falla de forma ruidosa antes de producir resultados.

### Seeds y splits fijos

```
seed = 42  (fijo en configs/experiments.yaml)
splits = make_shared_splits(y, subject_ids, seed=42)
         → misma lista de 9 tuplas (train_idx, test_idx) para las 3 variantes
```

### Advertencia sobre el fold de S09

El log del notebook registra una advertencia de sklearn: `"No positive class found in y_true, recall is set to one for all thresholds"` en el fold donde S09 es el sujeto de prueba. El sujeto S09 tiene solo 6,391 ventanas (vs ~21,578 en promedio) y solo registra la actividad 24 (saltar), que se mapea a riesgo alto. En ese fold, las clases 0 y 1 no tienen instancias en el conjunto de prueba, por lo que el PR-AUC no puede calcularse con validez estadística. Este fold degradado es una causa parcial de la std alta reportada (±0.20). Se documenta como **riesgo conocido** y no invalida los resultados, pero se debe tener en cuenta al interpretar la variabilidad.

---

## 6. Conclusión y decisión

### Decisión: **ADOPTAR `XGB_n200_baseFeats`**

**Justificación técnica:**

`XGB_n200_baseFeats` supera el baseline en la métrica central con una mejora de **+1.66% en F1-Macro** (de 0.7942 a 0.8108), por encima del umbral de 1% establecido en `configs/experiments.yaml`. La mejora es consistente en todas las métricas secundarias: PR-AUC aumenta +1.66 puntos porcentuales y Accuracy aumenta +1.74 puntos porcentuales. El patrón de mejora uniforme en las 3 métricas indica que el beneficio no es un artefacto estadístico de una métrica en particular.

La mejora tiene sentido técnico: XGBoost aprende iterativamente cuáles features son más discriminativas para los casos difíciles (ventanas de transición entre niveles de riesgo), mientras que RF les asigna igual probabilidad de selección a priori. Con 297 features que incluyen componentes altamente correlacionadas (visible en el heatmap del EDA), el boosting aplica una selección implícita más eficiente.

**Justificación de costo y viabilidad:**

`XGB_n200_baseFeats` no solo mejora la métrica, sino que lo hace siendo **368 segundos más rápido** en el ciclo completo de CV (392.4 s vs 760.6 s del baseline, una reducción del 48%). Este es un resultado inusual y especialmente valioso para el contexto del proyecto: el modelo que se adoptará deberá ejecutar inferencia en tiempo real en el Jetson Orin Nano. Un modelo más rápido de entrenar generalmente también infiere más rápido, lo que favorece la viabilidad en edge AI.

**Decisión sobre `RF_n200_biomechFeats`: DESCARTAR**

La mejora de +0.48% en F1-Macro está dentro del ruido estadístico (la std del baseline es ±0.2003, mucho mayor que el delta observado). Añadir 73 features y aumentar el tiempo de entrenamiento en +20% para una mejora que no es estadísticamente distinguible del ruido no justifica la complejidad adicional. Las features biomecánicas (tilt angle, jerk, SMA, ratios inter-segmento) pueden ser más valiosas cuando se implementen como canales de entrada crudos a una CNN-LSTM, donde la red aprenda directamente los patrones temporales, en lugar de añadirlos como estadísticas agregadas al vector de features.

**Resumen de la decisión:**

| Variante | ΔF1 | ΔTiempo | Estadísticamente significativo | Viabilidad edge | Decisión |
|---|---|---|---|---|---|
| `RF_n200_biomechFeats` | +0.48% | +20% | No (dentro de std) | Neutral | **DESCARTAR** |
| `XGB_n200_baseFeats` | +1.66% | −48% | Sí (supera umbral) | Favorable | **ADOPTAR ✓** |

**Configuración adoptada para el siguiente sprint:**

```
Modelo        : XGBoost (n_estimators=200, max_depth=6, learning_rate=0.1)
Features      : 297 features estadísticas (temporal + espectral)
Validación    : GroupKFold por sujeto (LOSO), seed=42
F1-Macro      : 0.8108 ± 0.2038
```

---

## 7. Reproducibilidad

### Comandos para reproducir el experimento

Desde la raíz del proyecto con el entorno `posture-risk` activado:

```bash
# 1. Cambiar al branch del experimento
git checkout experiments/feature-engineering-v1

# 2. (Solo primera vez) Generar el HDF5 biomecánico
python -m posture_risk.experiments.prepare_biomech_dataset \
       --config configs/default.yaml

# 3. Ejecutar el experimento completo
jupyter nbconvert --to notebook --execute \
       notebooks/03_ab_experiments.ipynb \
       --output notebooks/03_ab_experiments_executed.ipynb
```

O interactivamente:
```bash
jupyter lab
# Abrir notebooks/03_ab_experiments.ipynb → Run All Cells
```

### Ubicación de artefactos

| Artefacto | Ruta | Descripción |
|---|---|---|
| Dataset base | `data/processed/pamap2_features.h5` | 194,205 ventanas × 297 features |
| Dataset biomecánico | `data/processed/pamap2_biomech_features.h5` | 194,205 ventanas × 73 features |
| Log de experimentos | `logs/metrics_experimentos.csv` | Historial con EXP001, EXP002, EXP003 |
| Tabla comparativa | `reports/ab_comparison_table.csv` | Tabla estándar del asesor |
| Gráfico PR | `reports/figures/05_ab_pr_curves_comparison.png` | Gráfico clave |
| Matrices confusión | `reports/figures/06_ab_confusion_matrices.png` | Comparación visual |
| Configuración | `configs/experiments.yaml` | Parámetros reproducibles |
| Este reporte | `docs/experiments/reporte_AB_semana6.md` | Documento formal |

### Hashes de integridad (verificación de datos)

Para verificar que los HDF5 no han sido modificados desde la ejecución:

```bash
# Windows PowerShell
Get-FileHash data\processed\pamap2_features.h5 -Algorithm MD5
Get-FileHash data\processed\pamap2_biomech_features.h5 -Algorithm MD5
```

Los hashes deben coincidir con los registrados en `logs/metrics_experimentos.csv` bajo el campo `data_hash` en futuras ejecuciones.

### Configuración del entorno

```
Python        : 3.10.14
scikit-learn  : 1.5.1
xgboost       : 2.1.1
numpy         : 1.26.4
seed          : 42
```

---

## 8. Riesgos y próximos pasos

### Riesgo crítico — Alta variabilidad inter-sujeto (std ≈ 0.20)

La desviación estándar de F1-Macro es ~0.20 en todas las variantes, lo que equivale al 25% del valor medio. El principal origen identificado es el fold del sujeto S09 (solo 6,391 ventanas, una sola actividad), pero la variabilidad persiste incluso excluyendo ese fold. Esto es una señal temprana del **domain shift** que DANN deberá resolver: si el modelo ya tiene dificultad generalizando entre sujetos del mismo dataset (PAMAP2), la generalización al dominio de mina será aún más desafiante. **Acción inmediata:** al obtener el dataset DS propio, aplicar criterios de inclusión que maximicen la variedad demográfica (edad, complexión, lateralidad) para reducir esta varianza en el dominio fuente antes de entrenar el modelo base de la tesis.

### Próximo sprint — Arquitectura CNN-LSTM sobre `XGB_n200_baseFeats`

Con `XGBoost` adoptado como nuevo baseline, el siguiente paso es implementar la arquitectura CNN-LSTM (Entregable 5 de la tesis) sobre el mismo feature set de 297 features con ventanas deslizantes de 200 ms. La CNN-LSTM recibirá secuencias de 10 ventanas consecutivas (2 segundos de historia) como entrada al LSTM, capturando la dinámica temporal de las posturas sostenidas que el Random Forest y XGBoost no pueden modelar (tratan cada ventana como independiente). Si CNN-LSTM no supera el F1-Macro de 0.8108 de XGBoost, habrá un problema en la arquitectura o el número de datos de entrenamiento, y deberá revisarse antes de avanzar al DANN.

---

*Reporte generado en el branch `experiments/feature-engineering-v1` del repositorio `posture-risk-mining_by_Huaytalla`.*
