# Plan de Despliegue — Sistema de Detección Postural

**Proyecto:** Aprendizaje por Transferencia Optimizado para la Detección de Posturas Riesgosas de Extremidades Superiores en Operadores Perforistas en Unidades Mineras
**Autor:** Msc. Jaime Antonio Huaytalla Pariona
**Asesor:** Ing. Glen Rodríguez
**Sprint:** 7 — Empaquetado y despliegue mínimo
**Documento:** Plan de Despliegue (10 secciones)
**Formato:** Sesión 13 del asesor, slide 16

---

## 1. Resumen ejecutivo

El sistema **posture-risk** clasifica en tiempo real el nivel de riesgo ergonómico (bajo, medio, alto) de operadores perforistas mineros a partir de ventanas de 200 ms de señales IMU multimodales (27 canales × 100 Hz en muñeca, pecho y tobillo). El artefacto de este sprint es un **plan de despliegue ejecutable** para llevar el prototipo a un módulo Python distribuible.

**Arquitectura adoptada:** módulo CLI Python (`python -m posture_risk.predict`), coherente con el despliegue final en edge AI (Jetson Orin Nano) y con el flujo batch-analítico del contexto de tesis. El API REST queda en el roadmap.

**Modelo actual:** XGBoost con hiperparámetros optimizados (Sprint 3 HPO winner) + calibración isotónica + umbrales óptimos por clase con prioridad 1.5× a la clase "alto". F1-Macro LOSO = 0.8171 ± 0.2047 sobre PAMAP2 (9 sujetos, 194,205 ventanas). Baseline técnico: Random Forest (F1 = 0.7942). Baseline stakeholder: evaluación ergonómica manual RULA/REBA por observador humano.

**SLOs adoptados** (declarados en sprint 6, verificados aquí): latencia p95 < 200 ms, throughput ≥ 5 pred/s, tasa de errores < 1%, F1-Macro ≥ 0.80. Todos son críticos y bloquean despliegue si fallan.

**Contratos I/O** con Pydantic v2, cinco ejemplos JSON (válido, dos inválidos, dos de borde). **Reproducibilidad** con `environment.yml` (lockfile conda), Makefile con 5 targets (`setup, train, predict, test, e2e`) y su wrapper PowerShell equivalente. **Tests**: 7 smoke + 2 golden con tolerancia ±0.005 en probabilidades.

**Riesgos principales:** dependencia frágil en el orden de canales IMU (mitigación: validación estricta en Pydantic), degradación de calibración con cambio de dominio (mitigación: retrain trigger tras drift PSI > 0.2), y latencia impredecible en Jetson real vs medida en PC de desarrollo (mitigación: benchmark obligatorio pre-canary). **Rollback**: conservar `models/best_xgb_prev.pkl` etiquetado durante 30 días.

**Hoja de ruta a Docker/API:** 6 hitos programados entre sprints 8 y 12, con dockerización primero, luego adaptación ARM64/INT8 para Jetson, y finalmente API FastAPI + canary con voluntarios.

---

## 2. Arquitectura candidata

### 2.1 Decisión formal

**Arquitectura adoptada: Módulo CLI (Python)**

Se descarta la alternativa de Servicio (API REST con FastAPI/Flask) por las siguientes razones:

| Criterio | Módulo CLI ✓ | Servicio API |
|---|---|---|
| Coherencia con despliegue final (Jetson edge) | Alta — el operador consume alertas locales, no hay servidor | Baja — Jetson tendría que hostear API para consumo local, overhead innecesario |
| Complejidad operativa | Baja — un solo comando, sin servicios | Media — requiere gestión de proceso, healthchecks, gunicorn |
| Ideal para contexto tesis | Sí — batch, reproducible, sin red | Parcialmente — requiere infraestructura adicional |
| Integración con terceros | Vía archivos JSON/JSONL | Directa vía HTTP |
| Roadmap | Base para posterior wrapper API | — |

La regla del asesor (slide 8) resuelve el caso: *"Batch/analítica interna ↦ Módulo. Interacción online/terceros ↦ Servicio."* Este sistema es analítica edge sin servidor remoto en su fase actual.

### 2.2 Diagrama de arquitectura

Ver archivo `docs/deployment/architecture_diagram.mermaid`. Renderizado en Mermaid Live Editor:

```
Sensores IMU (100 Hz, 27 canales)
        ↓
Ventaneo (200 ms, 50% overlap)
        ↓
Extracción de 297 features
        ↓
┌─────────────────────────────────────────────┐
│ Módulo CLI: posture_risk.predict            │
│                                              │
│  1. Contratos I/O (Pydantic v2)             │
│  2. Preprocess                              │
│  3. Escalado (StandardScaler)               │
│  4. Inferencia (XGBoost HPO)                │
│  5. Calibración isotónica                   │
│  6. Umbral por clase (prioridad 1.5x Alto)  │
│  7. Salida: pred + probs + latencia + ver   │
└─────────────────────────────────────────────┘
        ↓                    ↓
Alerta al operador     Métricas + logs
   (Jetson)          (CSV + loguru estructurado)
```

### 2.3 Componentes del módulo

| Componente | Responsabilidad | Archivo |
|---|---|---|
| Contratos I/O | Validación Pydantic de entrada, salida, errores | `src/posture_risk/contracts/io_schemas.py` |
| Módulo CLI | Punto de entrada `python -m posture_risk.predict` | `src/posture_risk/predict.py` |
| Bundle de artefactos | Carga de modelo + calibradores + umbrales | dentro de `predict.py` |
| Pipeline por etapas | Ejecución ordenada de las 5 etapas | dentro de `predict.py` |
| Observabilidad | Logging estructurado + métricas CSV | dentro de `predict.py` |

---

## 3. Contratos I/O

### 3.1 Esquema de entrada — `PredictionInput`

```python
class PredictionInput(BaseModel):
    request_id:   str                     # UUID, 1-128 chars
    features:     List[float]             # 297 valores, sin NaN/Inf, en rango [-1e6, 1e6]
    subject_hint: Optional[int]           # opcional, ID de sujeto para trazabilidad
    timestamp:    Optional[datetime]      # opcional, momento de captura
```

**Batch:** `BatchPredictionInput` agrupa 1-10,000 `PredictionInput` con `request_id` únicos.

### 3.2 Esquema de salida — `PredictionOutput`

```python
class PredictionOutput(BaseModel):
    request_id:      str
    prediction:      RiskLevel            # "bajo" | "medio" | "alto"
    probabilities:   List[float]          # 3 valores, suman 1.0 ± 0.01
    confidence:      float                # max(probabilities)
    latency_ms:      float                # tiempo total de inferencia
    model_version:   str                  # SemVer: "v1.0.0"
    pipeline_stages: List[str]            # ["preprocess", "scale", "infer_xgb", ...]
    served_at:       datetime             # UTC
```

### 3.3 Esquema de error — `PredictionError`

```python
class PredictionError(BaseModel):
    request_id:  Optional[str]
    code:        ErrorCode                # enum: E001..E005, E999
    message:     str                      # descripción legible
    hint:        str                      # sugerencia de resolución
    field:       Optional[str]            # campo afectado si aplica
    timestamp:   datetime
```

**Códigos de error estables** (no cambian entre versiones):
- `E001_INVALID_INPUT` — falla validación de entrada
- `E002_MODEL_NOT_LOADED` — artefactos no encontrados
- `E003_INFERENCE_FAILED` — error durante inferencia
- `E004_TIMEOUT` — excedió el timeout de 5 segundos
- `E005_PAYLOAD_TOO_LARGE` — batch > 10,000 ventanas
- `E999_INTERNAL_ERROR` — error no clasificado

### 3.4 Ejemplos (5 casos)

Documentados en `contracts/examples.json`:

1. **Válido:** ventana normal con features en rango, request_id UUID, subject_hint presente
2. **Inválido (NaN):** features[42] = NaN → error E001 con hint sobre pipeline aguas arriba
3. **Inválido (longitud):** 296 en lugar de 297 features → error E001 sobre canal caído
4. **Borde (ambigüedad):** probabilidades ~equiprobables → predicción con confianza baja
5. **Borde (saturación):** features[15] en el límite superior → predicción válida + advertencia en logs

---

## 4. Reproducibilidad

### 4.1 Entorno

**Gestor:** conda con canal `conda-forge` para prioridad de versiones estables.
**Lockfile:** `configs/environment.yml` con versiones fijadas de todas las dependencias.
**Regeneración en máquina virgen:**

```bash
conda env create -f configs/environment.yml
conda activate posture-risk
python -m ipykernel install --user --name posture-risk --display-name "Python (posture-risk)"
```

### 4.2 Versiones críticas fijadas

| Paquete | Versión |
|---|---|
| Python | 3.10.14 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |
| xgboost | 2.1.1 |
| optuna | 3.6.1 |
| pydantic | 2.8.2 |
| loguru | 0.7.2 |
| pytest | 8.3.2 |

### 4.3 Seeds

Semilla global `SEED = 42`, configurada en:
- `configs/default.yaml` (parámetro global)
- `.env.example` (variable `SEED=42`)
- Cada uno de los pipelines de sklearn (`random_state=42`)
- XGBoost (`random_state=42`)
- StandardScaler (determinístico, no requiere seed)

### 4.4 Makefile (5 targets del asesor + auxiliares)

```
make setup    - Crear entorno conda desde environment.yml
make train    - Entrenar modelo + calibradores + umbrales, persistir a models/
make predict  - Ejecutar predicción sobre input de ejemplo
make test     - Ejecutar smoke + golden tests
make e2e      - Pipeline end-to-end en limpio
```

Auxiliares: `make lint` (validar ejemplos), `make clean` (limpiar artefactos), `make golden-regenerate` (regenerar dorado).

**Compatibilidad Windows:** el mismo interfaz se ofrece como `make.ps1` en PowerShell nativo, sin requerir GNU Make instalado. Ambos scripts leen `.env` automáticamente si existe.

### 4.5 Versionado de artefactos

- **Código:** git tags SemVer (`v1.0.0`, `v1.0.1`, etc.)
- **Modelo:** hash MD5 del archivo `models/best_xgb.pkl` calculado en carga y expuesto en cada `PredictionOutput.model_version`
- **Dataset:** hash MD5 del archivo `data/processed/pamap2_features.h5` verificado en `train_and_persist.py`

---

## 5. E2E en limpio

### 5.1 Escenario

Máquina virgen (Windows + Miniconda instalado). El objetivo es correr desde cero hasta obtener una predicción válida y tests verdes.

### 5.2 Pasos exactos

```bash
# 1. Clonar repositorio y entrar a la carpeta
git clone <repo-url> posture-risk-mining
cd posture-risk-mining
git checkout experiments/feature-engineering-v1

# 2. Configurar entorno
cp .env.example .env
make setup   # o .\make.ps1 setup en PowerShell

# 3. Activar entorno
conda activate posture-risk

# 4. Entrenar y persistir artefactos
make train   # o .\make.ps1 train

# 5. Generar archivo dorado (solo primera vez)
python -m tests.generate_golden

# 6. Ejecutar E2E completo
make e2e     # o .\make.ps1 e2e
```

### 5.3 Datos ejemplo

`data/examples/sample_input.json` — 5 ventanas seleccionadas de PAMAP2 con distribución de clases balanceada:
- 2 ventanas de riesgo bajo
- 2 ventanas de riesgo medio
- 1 ventana de riesgo alto

Generado automáticamente por `make train` desde el subset determinístico del dataset.

### 5.4 Criterios de éxito

El script `scripts/verify_e2e_output.py` valida:

1. Archivo `data/examples/e2e_output.json` existe y es JSON válido
2. Estructura `BatchPredictionOutput` completa (`n_processed`, `n_errors`, `predictions`, etc.)
3. `n_processed > 0` y `n_errors == 0`
4. Todas las predicciones tienen campos requeridos
5. Probabilidades suman 1.0 ± 0.01 y están en [0, 1]
6. Clase predicha ∈ {"bajo", "medio", "alto"}
7. Latencia máxima < 500 ms (umbral generoso para PC de desarrollo)

Adicionalmente, `make test` ejecuta:
- 7 smoke tests (pytest, deben pasar todos)
- 2 golden tests (comparación contra `tests/golden/golden_predictions.json` con tolerancia ±0.005)

Ver diagrama de flujo E2E en `docs/deployment/e2e_flow.mermaid`.

---

## 6. Observabilidad

### 6.1 Logging estructurado

Se usa `loguru` con formato personalizado que incluye:

```
{timestamp} | {level:<8} | {request_id} | {message}
```

Ejemplo:

```
2026-07-15 10:30:00.145 | INFO     | a3f8c9d2-... | Iniciando predicción de 5 ventanas
2026-07-15 10:30:00.157 | INFO     | a3f8c9d2-... | Predicción completa: 5 OK, 0 errores
```

**Nivel configurable** vía `.env` (variable `LOG_LEVEL=INFO`). Los logs van a STDERR por defecto para no contaminar la salida JSON en STDOUT.

### 6.2 Métricas locales (CSV)

Cada invocación del módulo escribe una fila en `logs/inference_metrics.csv`:

| Campo | Descripción |
|---|---|
| `timestamp` | UTC ISO 8601 |
| `batch_id` | ID del batch procesado |
| `n_processed` | Predicciones exitosas |
| `n_errors` | Errores |
| `total_latency_ms` | Latencia total del batch |
| `avg_latency_ms` | Latencia promedio por predicción |
| `model_version` | SemVer del modelo |
| `artifact_hash` | Primeros 12 chars del hash MD5 del artefacto |

Este CSV permite análisis offline de:
- Tendencia de latencia a lo largo del tiempo
- Detección de drift en probabilidades (junto con dumps periódicos de predicciones)
- Auditoría de qué versión del modelo produjo qué predicción

### 6.3 Endpoint `/healthz`

**No aplica** en este sprint porque la arquitectura adoptada es CLI, no API. Cuando se agregue la API en el roadmap (hito H4), se implementará el endpoint `/healthz` según patrón estándar:

```json
GET /healthz → 200 OK
{
  "status": "ok",
  "version": "v1.0.0",
  "model_hash": "a3f8c9d2ef01...",
  "uptime_s": 3600
}
```

### 6.4 Trazabilidad

Cada `request_id` está presente en:
- La entrada (`PredictionInput.request_id`)
- La salida (`PredictionOutput.request_id`)
- Los logs de loguru (via `.bind(request_id=...)`)
- El CSV (indirectamente vía `batch_id`)

Esto permite trazar cualquier predicción desde su ID único hasta el timestamp exacto de procesamiento.

---

## 7. Validación & tests

### 7.1 Validaciones de entrada (Pydantic)

- Rangos numéricos: features en `[-1e6, 1e6]`
- Nulos: NaN e Inf rechazados con índice del elemento problemático
- Tipos: `request_id: str`, `features: List[float]`, longitud exacta 297
- Unicidad: `request_id` únicos dentro de un `BatchPredictionInput`
- Pattern: `model_version` debe cumplir SemVer

Toda validación falla con un `ValidationError` de Pydantic que se traduce a `PredictionError` con `code=E001_INVALID_INPUT`.

### 7.2 Smoke tests (7 tests)

Archivo: `tests/test_smoke.py`

1. `test_valid_input_passes_validation` — input válido pasa
2. `test_input_with_nan_is_rejected` — NaN rechazado
3. `test_input_with_wrong_length_is_rejected` — longitud incorrecta rechazada
4. `test_input_with_infinity_is_rejected` — Inf rechazado
5. `test_valid_output_passes_validation` — salida válida pasa
6. `test_output_with_wrong_probability_sum_is_rejected` — probs no-normalizadas rechazadas
7. `test_output_with_invalid_model_version_is_rejected` — versión no-SemVer rechazada
8. `test_batch_with_duplicate_request_ids_rejected` — IDs duplicados rechazados

### 7.3 Golden tests (2 tests)

Archivo: `tests/test_golden.py`, dorado en `tests/golden/golden_predictions.json`.

**Selección de ventanas:** 10 índices deterministas distribuidos: `[100, 5000, 15000, 30000, 50000, 80000, 120000, 150000, 180000, 194000]`.

**Tolerancia:** predicciones idénticas (clase); probabilidades ±0.005.

**Regeneración:** solo tras cambio intencional del modelo mediante `make golden-regenerate` o `python -m tests.generate_golden`. Requiere commit explícito para que el cambio quede documentado.

**Comando de ejecución:**

```bash
make test          # smoke + golden
pytest tests/test_smoke.py -v      # solo smoke
pytest tests/test_golden.py -v     # solo golden
```

---

## 8. Seguridad & configuración

### 8.1 Variables de entorno

Plantilla en `.env.example` (versionada en git). Copia local `.env` (ignorada por git).

| Variable | Default | Propósito |
|---|---|---|
| `MODEL_PATH` | `models/best_xgb.pkl` | Ruta al pipeline sklearn+XGBoost |
| `CALIBRATORS_PATH` | `models/calibrators.pkl` | Calibradores isotónicos |
| `THRESHOLDS_PATH` | `models/thresholds.json` | Umbrales por clase |
| `METRICS_CSV` | `logs/inference_metrics.csv` | Métricas locales |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `MAX_PAYLOAD_WINDOWS` | `10000` | Máximo de ventanas por batch |
| `INFERENCE_TIMEOUT_S` | `5.0` | Timeout de inferencia |
| `SEED` | `42` | Semilla global |
| `MODEL_VERSION` | `v1.0.0` | Versión declarada del sistema |

### 8.2 Datos sensibles

- **PAMAP2:** dataset público anónimo (IDs numéricos 101-109). Sin PII.
- **Dataset propio DS (futuro):** anonimización obligatoria antes del entrenamiento; consentimiento informado y aprobación ética documentados.
- **Logs:** no incluyen datos crudos de sensores; solo IDs, timestamps y métricas agregadas.
- **Modelo entrenado:** no memoriza patrones biométricos identificables (verificado con ablación por sujeto en Sprint 5).

### 8.3 Límites operativos

| Límite | Valor | Fundamentación |
|---|---|---|
| `MAX_PAYLOAD_WINDOWS` | 10,000 | ~2000 segundos de operación; suficiente para un turno completo |
| `INFERENCE_TIMEOUT_S` | 5.0 | 25× el SLO p95 (200 ms); detecta cuelgues sin falsos positivos |
| Tamaño máximo del vector de features | 297 exactos | Validado por Pydantic |
| Rango de features | `[-1e6, 1e6]` | Cubre valores post-filtrado; detecta saturación de sensor |

### 8.4 Buenas prácticas de configuración

- Nunca hardcodear rutas, credenciales o parámetros en código
- `.env` en `.gitignore` desde el inicio del proyecto
- Cambios en `.env.example` requieren PR + revisión
- `python-dotenv` carga automática desde `.env` en cualquier script del proyecto

---

## 9. Hoja de ruta a Docker/API

### 9.1 Hitos programados

| Hito | Descripción | Responsable | Sprint | Fecha estimada |
|---|---|---|---|---|
| **H1** | Dockerización del módulo CLI (Dockerfile + build reproducible) | Msc. J. Huaytalla | Sprint 8 | Aug 2026 |
| **H2** | Implementación de API FastAPI con endpoints `/predict`, `/healthz`, `/version` | Msc. J. Huaytalla | Sprint 9 | Sep 2026 |
| **H3** | Adaptación a Jetson Orin Nano (ARM64, imagen base específica) | Msc. J. Huaytalla | Sprint 9 | Sep 2026 |
| **H4** | Cuantización INT8 + ONNX runtime para inferencia optimizada en Jetson | Msc. J. Huaytalla | Sprint 10 | Oct 2026 |
| **H5** | Integración con dataset propio DS (features + retraining) | Msc. J. Huaytalla | Sprint 11 | Nov 2026 |
| **H6** | Canary de despliegue: 3 voluntarios en laboratorio + monitoreo 2 semanas | Msc. J. Huaytalla | Sprint 12 | Dic 2026 |

### 9.2 Criterios de aceptación por hito

**H1 (Docker):**
- `docker build -t posture-risk:v1.0.0 .` funciona en máquina virgen
- Imagen final < 2 GB
- `docker run posture-risk:v1.0.0 predict --in ...` reproduce salida del módulo CLI

**H2 (API):**
- Endpoints `/predict` (POST), `/healthz` (GET), `/version` (GET)
- Latencia p95 API < 250 ms (200 ms de modelo + 50 ms de overhead HTTP)
- Documentación OpenAPI/Swagger autogenerada

**H3 (Jetson):**
- Imagen ARM64 corre en Jetson Orin Nano
- SLOs del sprint 6 verificados en Jetson real (no PC de desarrollo)

**H4 (Optimización):**
- Modelo cuantizado INT8 mantiene F1-Macro dentro de -1% del original
- Latencia p95 reducida al menos 30% vs modelo float32

**H5 (DS propio):**
- Pipeline de features de DS produce output compatible con `PredictionInput`
- Retraining sobre DS + PAMAP2 combinado no degrada F1 en LOSO

**H6 (Canary):**
- 3 voluntarios usan el sistema durante 2 semanas
- 0 incidentes críticos (fallas del sistema, alertas falsas > 20/hora)
- Evaluación cualitativa post-canary con encuesta de sesión 12

### 9.3 Dependencias entre hitos

```
H1 (Docker) → H2 (API) → H3 (Jetson)
                            ↓
                         H4 (INT8)
                            ↓
H5 (DS propio) ────────→ H6 (Canary)
```

H2 depende de H1 (Dockerfile es prerrequisito). H3 depende de H2 (necesita API para pruebas de integración). H4 depende de H3 (optimización solo tiene sentido en hardware objetivo). H6 depende de H4 y H5.

---

## 10. Riesgos & mitigaciones

### 10.1 Matriz de riesgos

| ID | Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|---|
| R1 | Cambio en orden de canales IMU (mano/pecho/tobillo) | Media | Alto | Validación estricta con Pydantic + fixture de canonicalización en pipeline |
| R2 | Degradación de calibración isotónica con drift de dominio | Alta | Alto | Monitoreo PSI/KS mensual + retrain trigger si PSI > 0.2 |
| R3 | Latencia impredecible en Jetson real vs PC de desarrollo | Alta | Medio | Benchmark obligatorio en Jetson antes de canary (hito H3) |
| R4 | Versiones de dependencias no reproducibles | Baja | Alto | Lockfile `environment.yml` + hash verificado en `train_and_persist.py` |
| R5 | Falla del artefacto en carga (`joblib` deserialización) | Baja | Alto | `E002_MODEL_NOT_LOADED` con hint accionable + fallback documentado |
| R6 | Payload gigante en producción agota memoria | Baja | Medio | `MAX_PAYLOAD_WINDOWS=10000` con error explícito `E005` |
| R7 | Timeout en inferencia por caso patológico | Baja | Bajo | `INFERENCE_TIMEOUT_S=5.0` + logging del caso para postmortem |
| R8 | Contaminación de `subject_id` como covariable (respondido en Sprint 5) | Nulo | — | Ya mitigado: `subject_hint` es opcional y NO usado por el modelo |

### 10.2 Rollback

**Política:** conservar el artefacto anterior etiquetado por 30 días.

**Convención de nombres:**
- `models/best_xgb.pkl` — versión activa
- `models/best_xgb_prev.pkl` — versión anterior (backup)
- `models/best_xgb_v1.0.0.pkl` — snapshots versionados (git LFS o almacenamiento externo)

**Procedimiento de rollback:**

```bash
# 1. Detener producción (o retirar el módulo de rotación)
# 2. Restaurar artefactos anteriores
mv models/best_xgb.pkl models/best_xgb_failed.pkl
cp models/best_xgb_prev.pkl models/best_xgb.pkl
cp models/calibrators_prev.pkl models/calibrators.pkl
cp models/thresholds_prev.json models/thresholds.json

# 3. Ejecutar E2E de verificación
make e2e

# 4. Si OK, retomar producción
# 5. Postmortem del fallo antes de re-desplegar la versión nueva
```

**Criterio de activación de rollback:**
- Cualquier SLO crítico falla en producción
- Golden test comienza a fallar en más del 30% de las predicciones
- Latencia p95 se dobla respecto al histórico
- Reporte de un usuario stakeholder crítico

### 10.3 Modos degradados

| Nivel | Descripción | Cuándo |
|---|---|---|
| **Nominal** | XGB HPO + calibración isotónica + umbrales por clase | Todo OK |
| **Sin postproceso** | XGB HPO + argmax directo (sin calibración ni umbrales) | Fallo del calibrador o umbrales corruptos |
| **Baseline** | Random Forest 200 árboles | Fallo total del modelo XGBoost adoptado |
| **CLI offline** | Módulo CLI sin API (si futura API cae) | Fallo del servicio HTTP en H2+ |

Cada nivel se activa manualmente por operador con configuración explícita (variable de entorno `DEGRADED_MODE=level_2`, por ejemplo). No hay activación automática — la política es "fail loudly, human decide next".

### 10.4 Recuperación tras drift

Si el monitoreo detecta drift (PSI > 0.2 en 5 features o más, sostenido por 7 días):

1. Alertar al equipo (email + issue en repositorio)
2. Ejecutar diagnóstico: qué features derivaron, qué slices están afectados (reusar herramientas del Sprint 5)
3. Decidir: recalibración post-hoc (rápida, sprint corto) vs retraining completo (largo, sprint completo)
4. Regenerar artefactos con `make train` + `make golden-regenerate` + `git commit`
5. Ejecutar E2E completo antes de desplegar
6. Actualizar `model_version` en SemVer (`v1.0.0 → v1.1.0` para recalibración, `→ v2.0.0` para retraining con dataset nuevo)

---

## Anexos

### A. Estructura de archivos entregada

```
sprint7/
├── Makefile                                # 5 targets del asesor
├── make.ps1                                # wrapper PowerShell
├── .env.example                            # plantilla de variables
├── configs/
│   └── environment.yml                     # lockfile conda
├── contracts/
│   ├── io_schemas.py                       # Pydantic v2
│   └── examples.json                       # 5 ejemplos (válido/inválido/borde)
├── scripts/
│   ├── predict.py                          # módulo CLI principal
│   ├── train_and_persist.py               # entrenar y guardar artefactos
│   ├── verify_e2e_output.py               # verificar salida E2E
│   └── validate_examples.py               # validar ejemplos contra esquemas
├── tests/
│   ├── test_smoke.py                       # 7 smoke tests
│   ├── test_golden.py                      # 2 golden tests
│   └── generate_golden.py                  # regenerar archivo dorado
└── docs/deployment/
    ├── plan_de_despliegue.md              # este documento
    ├── architecture_diagram.mermaid       # diagrama arquitectura
    └── e2e_flow.mermaid                    # diagrama flujo E2E
```

### B. Referencias

- Sesión 13 del asesor Ing. Glen Rodríguez, curso Proyecto de Investigación 2
- Documentación Pydantic v2: https://docs.pydantic.dev/2.8/
- Documentación conda: https://docs.conda.io/
- GNU Make manual: https://www.gnu.org/software/make/manual/
- Semver 2.0.0: https://semver.org/

---

*Documento generado en el sprint 7 como Plan de Despliegue formal según especificación del asesor. Los artefactos mínimos que respaldan el plan (código, tests, esquemas, Makefile, .env.example, ejemplos) están incluidos en el ZIP de entrega y listos para integración al repositorio de la tesis.*
