# Simulación del estudio con stakeholder

**Documento:** Prueba con usuario simulada (opción A — autoevaluación experta)
**Autor:** Jaime Antonio Huaytalla Pariona
**Modo de prueba adoptado:** A/B controlado + Usabilidad guiada simulada
**Fecha de la simulación:** Sprint 6

---

## 1. Nota metodológica

Este documento simula la prueba con stakeholder que idealmente se ejecutaría con operadores perforistas mineros reales. Dado que en la fase actual del proyecto no se dispone de acceso a personal minero en faena, la simulación se realiza mediante **autoevaluación experta** por parte del investigador, quien ha realizado revisión bibliográfica extensa sobre ergonomía minera y trabajado con casos de estudio de la industria durante el desarrollo del proyecto.

La autoevaluación experta es una técnica reconocida en fases tempranas de diseño de sistemas (Nielsen, 1994) cuando el acceso al usuario final es restringido, y se distingue de la evaluación real en dos aspectos:

- **Fortaleza:** el evaluador tiene visibilidad completa del sistema técnico y puede juzgar coherencia interna
- **Debilidad:** el evaluador no representa la perspectiva del usuario final (sesgo del constructor); los resultados deben validarse con usuarios reales antes del despliegue

Los resultados de esta simulación deben interpretarse como una **primera iteración de diseño**, no como validación con usuario final.

---

## 2. Modo de prueba adoptado (sesión 12)

El asesor lista tres modos de prueba con usuario:

| Modo | Descripción | Riesgo operativo | Información útil |
|---|---|---|---|
| Usabilidad guiada (lab) | Tareas + think-aloud | Bajo | Alta |
| A/B controlado | Variante A vs B | Bajo | Alta |
| Shadow mode | Corre en paralelo sin impactar decisiones | Muy bajo | Media |

**Modo adoptado:** combinación de **A/B controlado + usabilidad guiada** ambos simulados. El A/B se ejecuta cuantitativamente sobre las 194,205 ventanas del dataset PAMAP2 (notebook 08); la usabilidad guiada se ejecuta como autoevaluación por escenarios en este documento.

**Justificación:** en el estado actual del proyecto (sin acceso a operadores mineros, sin Jetson), estos dos modos son los que ofrecen la mayor información útil con el menor riesgo operativo posible.

---

## 3. Escenarios de prueba

Se diseñan tres escenarios representativos de la operación real de un perforista minero. Cada escenario se evalúa sobre las mismas ventanas para ambas variantes (baseline RF y modelo actual XGB + mitigaciones).

### 3.1 Escenario 1 — Caso típico: perforación estable

**Descripción operativa:** operador realizando perforación con martillo neumático en postura ergonómicamente neutra durante 30 segundos.

**Ventanas equivalentes en PAMAP2:** actividad `standing` sostenida (actividad 3) o `sitting` (actividad 2), sujeto en estado estable, sin transiciones.

**Métricas por evaluar:**
- Tasa de éxito por escenario (predicciones correctas / total de ventanas del escenario)
- Latencia percibida en el peor caso (p95 dentro del escenario)
- Consistencia de la predicción (varianza en probabilidades entre ventanas consecutivas)

### 3.2 Escenario 2 — Caso límite: transición postural

**Descripción operativa:** operador que transiciona entre postura neutra y postura riesgosa alta durante ~5 segundos (por ejemplo, inclinarse para reposicionar el martillo).

**Ventanas equivalentes en PAMAP2:** ventanas cerca de cambios de actividad (mask de transición del sprint 5).

**Métricas por evaluar:**
- Recall en el momento de riesgo (¿el sistema detecta el cambio a alto riesgo cuando ocurre?)
- Retraso en la detección (cuántas ventanas después de la transición se emite la alerta)
- Falsos positivos post-transición (si el modelo mantiene alerta después de que la postura vuelve a neutra)

### 3.3 Escenario 3 — Caso adversarial: actividad ambigua

**Descripción operativa:** operador realizando una micro-actividad que combina elementos de varias posturas en 200 ms (por ejemplo, un ajuste rápido de agarre con torsión leve del torso).

**Ventanas equivalentes en PAMAP2:** ventanas con etiqueta correcta pero baja confianza del modelo (probabilidad máxima < 0.6 en OOF).

**Métricas por evaluar:**
- Confianza del modelo en la predicción (¿se muestra dubitativo o hace afirmaciones fuertes erróneas?)
- Coherencia entre la clase predicha y el postproceso (¿la calibración isotónica ayuda a moderar la sobreconfianza?)

---

## 4. Autoevaluación experta

### 4.1 Escalas de calificación (sesión 12)

Se usan las tres escalas 1–5 definidas por el asesor:

- **Utilidad percibida (1-5):** 1 = el sistema no ayuda, 5 = el sistema mejora sustancialmente el proceso
- **Claridad de la salida (1-5):** 1 = no entiendo qué me dice, 5 = la salida es inequívoca
- **Confianza (1-5):** 1 = no confío en la recomendación, 5 = confío plenamente

### 4.2 Calificaciones — Modelo actual (XGB HPO + calibración + umbrales)

| Escenario | Utilidad | Claridad | Confianza | Comentario |
|---|:---:|:---:|:---:|---|
| 1 — Típico | 5 | 4 | 5 | Sistema clasifica correctamente y con alta confianza; salida binaria clara (bajo/medio/alto) |
| 2 — Límite | 4 | 4 | 4 | Detecta la transición pero con 1-2 ventanas de retraso; aceptable operativamente |
| 3 — Adversarial | 3 | 3 | 3 | Sobreconfianza detectada en casos ambiguos (heredado del hallazgo del sprint 5); la calibración isotónica mitiga parcialmente |
| **Media** | **4.0** | **3.7** | **4.0** | |

### 4.3 Calificaciones — Baseline RF

| Escenario | Utilidad | Claridad | Confianza | Comentario |
|---|:---:|:---:|:---:|---|
| 1 — Típico | 4 | 4 | 4 | Clasifica correctamente pero con menor confianza que el modelo actual |
| 2 — Límite | 2 | 3 | 2 | Retraso mayor en detectar transiciones; propenso a mantener la clase anterior |
| 3 — Adversarial | 3 | 3 | 3 | Similar al modelo actual, pero sin postproceso |
| **Media** | **3.0** | **3.3** | **3.0** | |

### 4.4 Diferencial (actual − baseline)

| Métrica | Baseline | Actual | Δ |
|---|:---:|:---:|:---:|
| Utilidad percibida | 3.0 | 4.0 | +1.0 |
| Claridad de la salida | 3.3 | 3.7 | +0.4 |
| Confianza | 3.0 | 4.0 | +1.0 |

**Interpretación:** el modelo actual mejora sensiblemente en utilidad y confianza percibida, especialmente en el escenario límite (transiciones). La claridad de la salida mejora poco porque el formato de la salida (tres niveles bajo/medio/alto) es idéntico entre ambas variantes.

---

## 5. Comentarios clave (citas simuladas)

Las siguientes citas representan verbalizaciones plausibles de un operador experto al usar el sistema durante los escenarios:

> **Escenario 1 (típico):** *"El sistema no me molesta cuando estoy en postura correcta. No hay alertas innecesarias, esto es lo que quiero."*

> **Escenario 2 (transición):** *"Detecta el cambio cuando me inclino, aunque tarda un segundo. Prefiero que tarde a que se equivoque."*

> **Escenario 3 (adversarial):** *"Cuando hago un movimiento raro me dice 'medio riesgo' con mucha seguridad y a veces no estoy de acuerdo. Ahí preferiría que se mostrara dubitativo."*

> **General:** *"Si esto funciona bien en la mina y no me pone a saltar alarmas cada rato, lo uso. Si me interrumpe mucho, lo apago al segundo día."*

---

## 6. Casos donde el usuario NO usaría la recomendación

Riesgos identificados:

1. **Ambiente ruidoso extremo** (perforación con martillo neumático a plena potencia): las vibraciones podrían saturar los sensores IMU y generar alertas incorrectas persistentes. Riesgo alto de desactivación por parte del operador.

2. **Tareas que requieren posturas técnicamente riesgosas por diseño** (por ejemplo, encajar el martillo en un agujero específico): el operador sabe que la postura es requerida y una alerta constante lo distrae. Riesgo medio de ignorar la alerta.

3. **Fatiga acumulada de fin de turno**: el operador ya está en postura degradada por cansancio y no tiene margen para corregir. Alertas insistentes generan frustración. Riesgo medio de desconfianza en el sistema.

4. **Interrupciones no clasificadas** (por ejemplo, ajustes de equipo): el sistema no distingue "trabajando en postura riesgosa" vs "arreglando el equipo temporalmente". Riesgo bajo pero relevante.

---

## 7. Guión ejecutado de la reunión con stakeholder (sesión 12)

### 7.1 Bloque 1 — Contexto y objetivo (2 min)

**Investigador:** *"Presentamos el sistema de detección de posturas ergonómicamente riesgosas en operadores perforistas. El objetivo es alertar en tiempo real cuando el operador entra en una postura que puede causar trastornos musculoesqueléticos crónicos. La decisión que afecta es si adoptamos este modelo como línea base para el próximo sprint de campo (Jetson + DS propio) o si volvemos al modelo anterior."*

**KPI de negocio:** tiempo de exposición a posturas de riesgo alto por hora de operación.

### 7.2 Bloque 2 — Demo (5-7 min)

**Investigador:** *"Ejecutamos dos casos: un operador en postura estable (caso típico) y un operador transicionando entre posturas (caso límite). En ambos casos, comparamos el sistema baseline con el actual."*

Se muestran los resultados del A/B controlado (notebook 08, sección 7): falsas alarmas/hora y recall de clase alto.

### 7.3 Bloque 3 — Tareas y escenarios (5-10 min)

**Investigador:** *"Aplicamos los 3 escenarios: típico, límite y adversarial. Le pido que evalúe cada uno en utilidad, claridad y confianza."*

Se registran las calificaciones de la sección 4.

### 7.4 Bloque 4 — Feedback estructurado (5-8 min)

Reflexiones simuladas del stakeholder:

- *"El sistema funciona en el caso típico y detecta transiciones con retraso mínimo."*
- *"Me preocupa el caso adversarial: el modelo se muestra muy seguro cuando debería dudar."*
- *"Las falsas alarmas por hora son aceptables si están bajo 5 por hora; sobre 10 sería inasumible."*
- *"La latencia bajo 200 ms es más que suficiente para no interferir con la operación."*

### 7.5 Bloque 5 — Cierre (2 min)

**Criterios de aceptación acordados:**

- Falsas alarmas < 10/hora en condiciones típicas ✓ (por confirmar tras notebook)
- Detección de riesgo alto con retraso máximo de 2 ventanas ✓ (por confirmar)
- Latencia p95 < 200 ms ✓ (por confirmar tras notebook)
- F1-Macro ≥ 0.80 ✓ (por confirmar)

**Próximos pasos:** validación en Jetson real (sprint futuro), integración con el dataset propio DS (sprint futuro), refinamiento del caso adversarial mediante CNN-LSTM (Entregable 5 de la tesis).

---

## 8. Manejo de feedback duro (sesión 12)

Se simulan tres feedbacks difíciles y las respuestas correspondientes:

**Feedback 1:** *"¿Y si el modelo se equivoca y no alerta al operador que está en riesgo alto?"*

**Respuesta:** *"Es un riesgo real. La métrica recall clase Alto que medimos cuantifica precisamente ese error. En el modelo actual con calibración y umbrales, el recall es X (por confirmar tras notebook). Además, adoptamos el SLO F1-Macro ≥ 0.80 como umbral crítico. Si se falla, no desplegamos. La mitigación adicional está en el postproceso: umbral por clase con prioridad 1.5x hacia alto riesgo, precisamente para minimizar este falso negativo."*

**Feedback 2:** *"La latencia me parece alta; en la mina esto tiene que responder de inmediato."*

**Respuesta:** *"Aceptamos el trade-off. Proponemos un mini-experimento: en el sprint siguiente medimos en Jetson Orin Nano real, y si la latencia p95 excede 200 ms aplicamos las optimizaciones adicionales (cuantización a INT8, ONNX runtime). El experimento tiene criterios claros y timeline de 1-2 semanas."*

**Feedback 3:** *"¿Cómo sé que este modelo funcionará con operadores nuevos que no estaban en el dataset?"*

**Respuesta:** *"Muy buena pregunta. Toda nuestra evaluación se hace con Leave-One-Subject-Out cross-validation, que es exactamente 'entrenar sin el operador X, probar con el operador X'. La std de 0.20 en el F1-Macro que reportamos viene precisamente de esta prueba. Además, el próximo sprint incorpora DANN (Domain Adversarial Neural Network) para adaptar el modelo al dominio real de la mina sin necesidad de anotar cada operador nuevo."*

---

## 9. Conclusiones de la simulación

**C1.** El modelo actual mejora sobre el baseline RF en las tres métricas de percepción de usuario (utilidad +1.0, claridad +0.4, confianza +1.0), con la mejora más marcada en escenarios de transición postural.

**C2.** El caso adversarial (Escenario 3) es el que menor calificación recibe en ambas variantes. La sobreconfianza en casos ambiguos, ya identificada en el sprint 5, sigue siendo un punto débil que se atenderá con arquitecturas más expresivas en el trabajo futuro (CNN-LSTM).

**C3.** Los riesgos operativos identificados (ambiente ruidoso, tareas por diseño riesgosas, fatiga de fin de turno) definen los límites de aplicabilidad del sistema y deben documentarse en el manual de usuario.

**C4.** Los criterios de aceptación acordados en el guión de reunión están alineados con los SLOs formalmente declarados; esto refuerza la coherencia entre expectativa del stakeholder y validación técnica.

---

