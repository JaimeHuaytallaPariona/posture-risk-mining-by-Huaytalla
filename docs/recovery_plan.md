# Plan de Recuperación en Ejecución

**Proyecto:** Detección de Posturas Riesgosas en Operadores Perforistas mediante Transfer Learning  
**Versión:** 1.0 | **Fecha:** 2025 | **Responsable:** [Tu nombre]

---

## Matriz de riesgos

| ID | Riesgo | Probabilidad | Impacto | Nivel | Estado |
|---|---|---|---|---|---|
| R01 | Retraso comité de ética | Alta | Crítico | 🔴 Alto | ✅ Mitigado |
| R02 | Negativa empresa minera | Media | Crítico | 🔴 Alto | 🟡 En mitigación |
| R03 | Fallo de hardware en campo | Media | Alto | 🟠 Medio | ✅ Mitigado |
| R04 | Desbalance severo de clases | Alta | Alto | 🟠 Medio | ✅ Mitigado |
| R05 | Inestabilidad entrenamiento DANN | Alta | Alto | 🟠 Medio | 🟡 En mitigación |
| R06 | Sincronización deficiente sensores | Media | Alto | 🟠 Medio | 🟡 En mitigación |
| R07 | Poca variabilidad en DS | Media | Medio | 🟡 Bajo | ⏳ Pendiente |

---

## R01 — Retraso del comité de ética

**Impacto:** Bloquea Entregables 3 y 6.

**Acciones en ejecución:**
- ✅ Pipeline completo validado sobre PAMAP2 (no requiere aprobación ética)
- ✅ Tiempo de espera usado productivamente: Entregables 1, 2, 4 y 5 avanzan sobre datos públicos
- ✅ Protocolo redactado con referencias a estudios publicados para reducir revisiones

---

## R02 — Negativa o demora de la empresa minera

**Impacto:** Imposibilita el Entregable 6 (Dt, datos de campo reales).

**Acciones en ejecución:**
- ✅ Carta formal de intención enviada a empresa minera
- 🟡 Lista de 5 voluntarios alternativos (estudiantes de Ing. de Minas) identificados
- 🟡 Protocolo de "laboratorio enriquecido" redactado: perforadora simulada con carga equivalente

---

## R03 — Fallo de hardware en campo

**Impacto:** Sesión de grabación inválida.

**Acciones en ejecución:**
- ✅ 2 nodos ESP32-C3 de repuesto ensamblados y calibrados
- ✅ Protocolo de calibración documentado en docs/sensor_placement.md
- ✅ Software detecta desconexión y marca gap sin invalidar el stream completo
- ✅ Fallback: grabación local en SD card de cada nodo

---

## R04 — Desbalance severo de clases

**Impacto:** El modelo aprende a predecir siempre la clase mayoritaria.

**Acciones en ejecución:**
- ✅ Métrica principal: F1-Macro (insensible al desbalance)
- ✅ `class_weight="balanced"` en todos los modelos
- ✅ SMOTE implementado como alternativa en el pipeline
- ✅ EDA cuantifica desbalance explícitamente (notebook 01)

---

## R05 — Inestabilidad del entrenamiento DANN

**Impacto:** Entregables 7 y 8 no convergen.

**Acciones en ejecución:**
- 🟡 Schedule progresivo de λ implementado (Ganin et al., 2016)
- 🟡 MLflow configurado para tracking automático de cada experimento
- 🟡 Fallback: Maximum Mean Discrepancy (MMD) como alternativa estable

---

## R06 — Sincronización deficiente entre sensores

**Impacto:** Ángulos RULA calculados incorrectamente (error > 10 ms invalida la correlación IMU–cámara).

**Acciones en ejecución:**
- ✅ Router Wi-Fi dedicado con servidor NTP local
- ✅ Software registra timestamp de cada paquete y calcula drift inter-sensor
- 🟡 Pruebas de sincronización con señal de referencia programadas

---

## Cronograma con contingencias

| Semana | Actividad | Plan B |
|---|---|---|
| 1–4 | Aprobación ética + permisos empresa | PAMAP2 + voluntarios universitarios |
| 5–8 | Grabaciones laboratorio (DS) | Ampliar a 20 voluntarios |
| 9–10 | Procesamiento y etiquetado RULA | Revisión manual del 10% de muestra |
| 11–14 | CNN-LSTM base | RF baseline si CNN-LSTM no converge |
| 15–18 | Trabajo de campo (Dt) | Laboratorio enriquecido |
| 19–22 | DANN | MMD como fallback |
| 23–24 | Escritura y defensa | — |

---

*Revisado mensualmente. Última actualización: 2025.*
