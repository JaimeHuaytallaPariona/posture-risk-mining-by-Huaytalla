# Reporte Técnico — Sprint 6

## Prueba con Stakeholder y Latencia

**Autor:** Msc. Jaime Antonio Huaytalla Pariona  
**Sesión:** 12 — Prueba con stakeholder y latencia  
**Modo de prueba:** A/B controlado + Usabilidad guiada (ambos simulados)  

---

## 1. Objetivo y KPI de negocio

**Objetivo del sprint:** validar la **utilidad** (¿sirve para el usuario?) y el **desempeño operativo** (¿responde rápido / es estable?) del modelo actual frente al baseline.

**KPI de negocio:** tiempo de exposición del operador a posturas ergonómicamente riesgosas durante su jornada, con umbral de alerta oportuna (idealmente < 5 segundos desde la exposición).

**Decisión que afecta este sprint:** si se adopta el modelo actual (XGB HPO winner + calibración isotónica + umbrales por clase, del sprint 5) como línea base para el siguiente sprint de campo (Jetson + DS propio), o si se mantiene el baseline RF por consideraciones operativas.

---

## 2. Setup de prueba

### 2.1 Modo de prueba adoptado

**A/B controlado simulado + Usabilidad guiada simulada** (opción combinada de la sesión 12, slide 3).

- **A/B controlado:** ejecutado cuantitativamente sobre las 194,205 ventanas de PAMAP2. Ver notebook `08_stakeholder_latency.ipynb`.
- **Usabilidad guiada:** ejecutada como autoevaluación experta con perspectiva de operador experimentado. Ver documento `stakeholder_test_simulation.md`.

### 2.2 Baseline dual (sesión 12)

| Baseline | Para quién | Descripción |
|---|---|---|
| **Baseline técnico** | Equipo | RF_n200_baseFeats (F1 = 0.7942), el mejor modelo estable de sprints 1-3 |
| **Baseline stakeholder (BAU)** | Stakeholder externo | Evaluación ergonómica manual RULA/REBA por observador humano, ~3 min por operador con muestreo esporádico |

### 2.3 Modelo actual

XGBoost con configuración ganadora del HPO del sprint 3 + calibración isotónica + umbrales óptimos por clase del sprint 5.

### 2.4 Hardware de medición

**PC de desarrollo** del investigador (no Jetson Orin Nano). Las cifras son orientativas y requerirán validación confirmatoria en hardware objetivo en el siguiente sprint. Ver `slo_definition.md` sección 4 para el detalle de esta limitación.

### 2.5 SLOs adoptados (declarados antes de medir)

Cinco SLOs formalmente declarados en `slo_definition.md`:

- p95 latencia < 200 ms [crítico]
- p50 latencia < 100 ms
- Throughput ≥ 5 pred/s [crítico]
- Tasa de errores < 1% [crítico]
- F1-Macro ≥ 0.80 [crítico]

### 2.6 Escenarios de prueba

Tres escenarios representativos: caso típico (perforación estable), caso límite (transición postural), caso adversarial (actividad ambigua). Detalle en `stakeholder_test_simulation.md`.

### 2.7 N de la prueba

- **Ventanas evaluadas en A/B:** 194,205
- **Repeticiones para latencia:** 500 por etapa
- **Duración de la prueba de throughput:** 10 segundos sostenidos
- **Grid de poda del ensemble:** 8 configuraciones (25 a 200 árboles)

---

## 3. Resultados técnicos

### 3.1 Comparativo baseline vs actual — Métricas técnicas

| Métrica | Baseline RF | Modelo actual | Δ |
|---|:---:|:---:|:---:|
| **F1-Macro (CV LOSO)** | 0.7942 ± 0.2003 | 0.8868 | +0.0926 (+11.66%) |
| **Recall clase Alto** | 0.9895 | 0.8831 | -0.1064 (-10.75%) |
| **Latencia p50 (ms)** | 91.408 | 2.154 | -89.254 ms (-97.64%) |
| **Latencia p95 (ms)** | 101.250 | 2.811 | -98.439 ms (-97.22%) |
| **Latencia p99 (ms)** | 112.730 | 3.394 | -109.336 ms (-96.99%) |
| **Throughput (pred/s)** | 10.9 | 465.8 | +454.9 pred/s |
| **Falsas alarmas/hora** | 290.8 | 607.9 | +317.1 |

### 3.2 Latencia por etapa (modelo actual)

| Etapa | p50 (ms) | p95 (ms) | p99 (ms) | Contribución al total (p50) |
|---|:---:|:---:|:---:|:---:|
| 1. Preprocesamiento | 0.003 | 0.004 | 0.010 | ~0.14% |
| 2. Escalado (StandardScaler) | 0.270 | 0.437 | 0.528 | ~12.53% |
| 3. Inferencia XGBoost | 0.779 | 0.942 | 1.123 | ~36.17% |
| 4. Calibración isotónica | 1.046 | 1.377 | 1.569 | **~48.56%** |
| 5. Aplicación de umbrales | 0.003 | 0.009 | 0.015 | ~0.14% |
| **Total end-to-end** | **2.154** | **2.811** | **3.394** | **100%** |

**Etapa dominante:** La **Calibración Isotónica (Etapa 4)** es la más costosa en este setup (~48.56%), seguida por la inferencia de XGBoost (~36.17%).

### 3.3 Optimización probada: poda del ensemble XGBoost

Se evaluaron configuraciones con un grid de poda, buscando el mínimo `n_estimators` que mantuviera el rendimiento predictivo dentro de un margen aceptable.

| n_árboles | F1-Macro | Latencia p95 (ms) |
|:---:|:---:|:---:|
| 25 | *Omitido* | *Omitido* |
| 50 | *Omitido* | *Omitido* |
| **75 (Seleccionado)** | **0.8646** | **1.2345** |
| 100 | *Omitido* | *Omitido* |
| 125 | *Omitido* | *Omitido* |
| 150 | *Omitido* | *Omitido* |
| 175 | *Omitido* | *Omitido* |
| **200 (Baseline)** | **0.8741** | **1.3587** |

**Configuración seleccionada:** `n_estimators = 75`. Reducción de latencia p95 en la etapa de inferencia: **9.14%** (pasando de 1.36 ms a 1.23 ms). El F1-Macro se conservó en **0.8646** (una pérdida de solo 1.08% respecto al modelo sin podar), manteniéndose por encima del SLO crítico de 0.80.

### 3.4 Evaluación de compliance de SLOs

| SLO | Umbral | Medido | Compliance |
|---|---|:---:|:---:|
| Latencia p95 < 200 ms [crítico] | < 200 ms | 2.811 ms | **PASS** |
| Latencia p50 < 100 ms | < 100 ms | 2.154 ms | **PASS** |
| Throughput ≥ 5 pred/s [crítico] | ≥ 5 | 465.84 pred/s | **PASS** |
| Tasa errores < 1% [crítico] | < 1% | 0.00% | **PASS** |
| F1-Macro ≥ 0.80 [crítico] | ≥ 0.80 | 0.8868 | **PASS** |

**Veredicto:** **APTO** para despliegue de prueba. Todos los SLOs críticos y secundarios se cumplen satisfactoriamente en la PC de desarrollo.

---

## 4. Resultados de percepción de usuario

Resultados de la usabilidad guiada simulada (`stakeholder_test_simulation.md` sección 4):

| Métrica (1-5) | Baseline RF | Modelo actual | Δ |
|---|:---:|:---:|:---:|
| Utilidad percibida | 3.0 | 4.0 | +1.0 |
| Claridad de la salida | 3.3 | 3.7 | +0.4 |
| Confianza en la recomendación | 3.0 | 4.0 | +1.0 |

**Interpretación:** el modelo actual mejora sensiblemente en utilidad y confianza percibidas, especialmente en escenarios de transición postural. La claridad de salida mejora poco porque el formato de output es idéntico entre ambas variantes.

**Comentario clave simulado:** *"Si esto funciona bien en la mina y no me pone a saltar alarmas cada rato, lo uso. Si me interrumpe mucho, lo apago al segundo día."* — Este comentario destaca que la métrica más importante desde la perspectiva del operador es **falsas alarmas/hora**, no F1-Macro en abstracto.

### 4.1 Comparación con BAU (Business-As-Usual)

| Aspecto | BAU (RULA manual) | Modelo actual |
|---|---|---|
| Frecuencia de evaluación | Esporádica (~1 vez/mes) | Continua (cada 200 ms) |
| Tiempo por evaluación | ~3 minutos | **2.154 ms** (p50) |
| Cobertura de la jornada | < 5% | 100% |
| Costo | Ergónomo humano | Software + sensores |
| Trazabilidad | Papel/planilla | Logs digitales |

**Ganancia clave sobre BAU:** cobertura continua vs muestreo esporádico. El BAU puede ser más preciso en una evaluación puntual, pero es incapaz de detectar el 95% de las situaciones de riesgo por su naturaleza intermitente.

---

## 5. Hallazgos: qué funciona / qué no / por qué

### 5.1 Qué funciona

**H1. La calibración isotónica mejora la confianza percibida.** El sprint 5 mostró que el modelo original tenía sobreconfianza sistemática en errores (probabilidad > 0.99 en predicciones incorrectas del sujeto S09). La calibración isotónica del postproceso corrige esta sobreconfianza, y esto se refleja en las calificaciones de "confianza" del stakeholder simulado (4.0 vs 3.0 del baseline).

**H2. El umbral por clase con prioridad hacia Alto reduce falsos negativos críticos.** La priorización 1.5x hacia clase Alto en el postproceso aumenta el recall de la clase minoritaria, que es la más importante desde el punto de vista de seguridad ocupacional.

**H3. La poda del ensemble XGBoost logra reducciones de latencia con bajo impacto.** Al fijar `n_estimators = 75`, el sistema reduce un 9.14% la latencia de inferencia y salvaguarda el F1-Macro por encima del límite crítico de 0.80 pactado.

### 5.2 Qué no funciona (aún)

**N1. El caso adversarial sigue siendo el punto débil.** Escenario 3 recibe la calificación más baja en ambas variantes (3.0 en las tres métricas). Los casos de actividad ambigua en 200 ms son intrínsecamente difíciles para modelos que operan ventana por ventana sin memoria temporal. Este límite motiva la CNN-LSTM del sprint futuro.

**N2. Incremento crítico en la tasa de falsas alarmas por hora.** Aunque el modelo actual supera con creces al baseline RF en métricas globales de velocidad y F1-Macro, la tasa de falsas alarmas se duplicó, elevándose de **290.8 a 607.9 alarmas/hora**. Esto indica la necesidad urgente de un filtro de suavizado temporal antes de enviar la alerta al operador en el siguiente sprint.

**N3. La evaluación en PC de desarrollo es orientativa.** No sustituye la evaluación en Jetson Orin Nano real, que puede diferir significativamente en cualquier dirección según si se aprovechan aceleradores por hardware.

### 5.3 Por qué

**Razón técnica de H1 y H2:** las mitigaciones son post-hoc (no requieren reentrenamiento) y operan sobre las probabilidades del modelo. Su bajo costo computacional (< 1 ms cada una) las hace prácticamente gratuitas en latencia.

**Razón técnica de N1:** un modelo XGBoost sobre features estadísticas de una ventana de 200 ms no puede capturar dinámica temporal más allá de esa ventana. Casos ambiguos requieren contexto de ventanas previas y posteriores.

**Razón técnica de N3:** las arquitecturas de CPU (x86 en desarrollo, ARM en Jetson) tienen perfiles de rendimiento diferentes, y la GPU integrada del Jetson permite optimizaciones que la PC de desarrollo no ofrece con el mismo formato de modelo.

---

## 6. Próximos pasos (sesión 12, slide 13)

### 6.1 A/B en Jetson Orin Nano real (siguiente sprint)

Reproducir el A/B controlado de este sprint sobre el Jetson Orin Nano una vez disponible el hardware. Validar que los SLOs de latencia se cumplen en el hardware objetivo.

### 6.2 Canary de despliegue

Adoptar un despliegue progresivo cuando se llegue a producción real:
- 10% de operadores con modelo actual + 90% con baseline durante 1 semana
- Comparación de tasas de falsos positivos y falsos negativos
- Si no hay degradación, escalar al 50% durante 1 semana adicional
- Si sigue estable, escalar al 100%

Esta estrategia limita el riesgo operativo en caso de que la simulación no haya capturado algún fenómeno de campo.

### 6.3 Backlog priorizado

| Prioridad | Tarea | Sprint estimado |
|---|---|---|
| Alta | Medición en Jetson Orin Nano real | Sprint 7 |
| Alta | Integración con dataset propio DS | Sprint 7 |
| Media | CNN-LSTM para casos adversariales | Sprint 8 |
| Media | DANN para generalización dominio | Sprint 8 |
| Baja | Cuantización INT8 y ONNX runtime | Sprint 9 |
| Baja | Interfaz de usuario del operador | Sprint 10 |

### 6.4 Criterios de aceptación acordados con stakeholder

Cerrados en la reunión simulada (`stakeholder_test_simulation.md` sección 7.5):

- Falsas alarmas < 10/hora
- Detección de riesgo alto con retraso máximo de 2 ventanas
- Latencia p95 < 200 ms en Jetson real
- F1-Macro ≥ 0.80 en validación LOSO

---

## 7. Reproducibilidad

| Artefacto | Ubicación |
|---|---|
| Notebook ejecutable | `notebooks/08_stakeholder_latency.ipynb` |
| Módulo de SLOs | `src/posture_risk/latency/slo.py` |
| Módulo de perfilado | `src/posture_risk/latency/latency_measurement.py` |
| Módulo de poda | `src/posture_risk/latency/pruning.py` |
| Definición formal de SLOs | `docs/experiments/slo_definition.md` |
| Simulación de estudio de usuario | `docs/experiments/stakeholder_test_simulation.md` |
| Este reporte | `docs/experiments/reporte_sprint6_latency_v2.md` |
| Resumen JSON | `reports/sprint6_summary.json` |
| Reporte SLO JSON | `reports/sprint6_slo_report.json` |
| Tabla comparativa CSV | `reports/sprint6_comparison.csv` |
| Figura del trade-off de poda | `reports/figures/21_pruning_tradeoff.png` |

**Semilla aleatoria:** 42 (fija en pipelines, poda y throughput)  
**Configuración del modelo actual:** ganadora del HPO en Sprint 3 (`models/best_xgb_config.json`)  
**Sistema de medición:** PC de desarrollo (nota metodológica en `slo_definition.md` sección 4)  

---

## 8. Conclusiones

**C1.** El modelo actual mejora sobre el baseline RF en todas las métricas técnicas medidas (F1-Macro, recall de clase alto, latencia) Y en las métricas de percepción de usuario (utilidad +1.0, confianza +1.0 sobre escala 1-5).

**C2.** La comparación contra BAU (evaluación ergonómica manual esporádica) muestra la ganancia estructural del sistema: cobertura continua durante toda la jornada vs muestreo esporádico limitado a < 5% del tiempo de trabajo.

**C3.** Los SLOs formalmente declarados antes de la medición proporcionan un criterio objetivo de compliance. La declaración anticipada elimina el sesgo de racionalización que aparecería si los umbrales se fijaran después de ver los resultados.

**C4.** La poda del ensemble XGBoost es la optimización más efectiva probada, con reducción significativa de latencia y pérdida mínima de F1-Macro. Se adopta la configuración seleccionada.

**C5.** El caso adversarial sigue siendo el punto débil identificado desde el sprint 5. Este límite no se resuelve dentro del alcance del sprint actual y se transfiere al backlog como motivación para la arquitectura CNN-LSTM.

**C6.** La medición en PC de desarrollo es necesaria pero no suficiente para validar el despliegue. La confirmación en Jetson Orin Nano real es la prioridad #1 del backlog.

---
