# Adenda al Plan de Despliegue — Correcciones post-ejecución

**Documento base:** `docs/deployment/plan_de_despliegue.md`
**Motivo:** Documentar dos ajustes aplicados durante la ejecución del sprint 7
**Fecha:** Sprint 7, cierre

---

## Correcciones aplicadas

Durante la ejecución del pipeline E2E se identificaron dos ajustes al plan original. Ambos fueron resueltos y están presentes en el código y estructura del repositorio, pero conviene documentarlos formalmente para trazabilidad.

### Corrección 1 — Script auxiliar `generate_sample_input.py`

**Contexto:** el pipeline de reproducibilidad de la sección 4 del plan definía cinco targets (`setup, train, predict, test, e2e`) y un artefacto `sample_windows.npz` generado por `train_and_persist.py`. Sin embargo, faltaba el paso intermedio que convierte el sample en formato `.npz` (numpy) al formato `.json` que espera el módulo CLI de predicción.

**Solución aplicada:** se agregó el script `scripts/generate_sample_input.py` que toma el sample de `.npz`, selecciona 5 ventanas con distribución balanceada de clases (2 bajo + 2 medio + 1 alto), y produce el archivo `data/examples/sample_input.json` compatible con el esquema `BatchPredictionInput`.

**Actualización al flujo E2E de la sección 5:** el orden correcto de comandos en máquina virgen es:

```powershell
.\make.ps1 setup
.\make.ps1 train
python scripts\generate_sample_input.py    # ← paso añadido
python -m tests.generate_golden
.\make.ps1 e2e
```

**Racional metodológico:** mantener `generate_sample_input.py` como script independiente (no dentro del target `train`) preserva la trazabilidad del sprint. Un revisor externo puede auditar cómo se selecciona el sample de prueba sin adentrarse en el pipeline de entrenamiento.

### Corrección 2 — Archivo `.gitignore`

**Contexto:** la sección 4 del plan mencionaba la separación entre código versionable y artefactos regenerables, pero no formalizaba qué archivos concretos deben quedar excluidos del control de versiones. Durante el primer intento de push a GitHub, el archivo `sample_windows.npz` (146 MB) excedió el límite de 100 MB del servicio, bloqueando la subida.

**Solución aplicada:** se agregó `.gitignore` con exclusiones explícitas de los siguientes artefactos:

| Categoría | Archivos excluidos | Regeneración |
|---|---|---|
| Datos derivados | `data/examples/sample_windows.npz` | `make train` |
| Inputs generados | `data/examples/sample_input.json`, `data/examples/e2e_output.json` | `generate_sample_input.py` y `make predict` |
| Modelos entrenados | `models/best_xgb.pkl`, `models/calibrators.pkl`, `models/thresholds.json` | `make train` |
| Logs de invocación | `logs/inference_metrics.csv`, `logs/*.log` | Se generan al invocar el módulo |
| Configuración local | `.env` | Copiar desde `.env.example` |
| Cachés Python | `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ipynb_checkpoints/` | Automáticas |

**Archivo versionado excepcionalmente:** `tests/golden/golden_predictions.json` (~1 KB). Este archivo sí forma parte del repositorio porque el golden test lo requiere como referencia. Su tamaño es despreciable y su versionado garantiza reproducibilidad del test entre desarrolladores.

**Actualización al presupuesto de reproducibilidad:** un usuario que clona el repositorio desde cero necesita ejecutar la secuencia completa `setup → train → generate_sample_input → generate_golden → e2e` para tener todos los artefactos listos. Esta secuencia toma aproximadamente 10 minutos en una PC de desarrollo estándar.

---

## Nota sobre el flujo de "máquina virgen" actualizado

La sección 5 del plan describe el objetivo de correr E2E en máquina virgen. Con las correcciones anteriores, el flujo actualizado es:

```powershell
# 1. Clonar repositorio
git clone https://github.com/JaimeHuaytallaPariona/posture-risk-mining-by-Huaytalla.git
cd posture-risk-mining-by-Huaytalla
git checkout experiments/feature-engineering-v1

# 2. Configurar entorno (política PowerShell + variables)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
copy .env.example .env

# 3. Instalar dependencias
.\make.ps1 setup
conda activate posture-risk

# 4. Entrenar modelo y generar artefactos
.\make.ps1 train

# 5. Generar sample de entrada
python scripts\generate_sample_input.py

# 6. Generar archivo dorado (una sola vez)
python -m tests.generate_golden

# 7. Ejecutar pipeline E2E completo
.\make.ps1 e2e
```

**Duración total estimada:** ~10 minutos, dependiendo de la velocidad de descarga de dependencias en el paso 3 (~5 min) y el entrenamiento del modelo en el paso 4 (~3 min).

**Criterio de éxito:** el paso 7 termina con `PIPELINE COMPLETADO EXITOSAMENTE` y sin errores en los tests.

---

## Impacto sobre los 10 puntos del plan original

Ninguna de las dos correcciones altera la arquitectura, los contratos I/O, los SLOs, los tests, los riesgos o el roadmap. Ambas son **complementos operativos** que hacen el plan efectivamente ejecutable en máquina virgen.

El resumen ejecutivo, la arquitectura candidata, los contratos I/O, los SLOs, las mitigaciones, y el roadmap H1-H6 permanecen sin cambios.

---

*Adenda generada tras la ejecución exitosa del sprint 7 en el entorno del investigador. Documenta ajustes operativos identificados durante la puesta en marcha del pipeline en máquina real, sin modificar el diseño ni el alcance del plan original.*
