# =============================================================================
# Makefile — Sprint 7: Empaquetado y despliegue mínimo
# Sesión 13 del asesor (slide 10): 5 targets mínimos
#   make setup | train | predict | test | e2e
#
# Requiere GNU Make. En Windows funciona con Git Bash o WSL.
# Alternativa PowerShell nativa: usar make.ps1
# =============================================================================

.PHONY: help setup train predict test e2e clean lint golden-regenerate all

# Detección de OS para adaptar comandos
ifeq ($(OS),Windows_NT)
    PYTHON := python
    RM := del /Q
    MKDIR := mkdir
else
    PYTHON := python3
    RM := rm -f
    MKDIR := mkdir -p
endif

# Cargar variables desde .env si existe
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Rutas por defecto (sobreescribibles vía variables de entorno)
MODEL_PATH        ?= models/best_xgb.pkl
CALIBRATORS_PATH  ?= models/calibrators.pkl
THRESHOLDS_PATH   ?= models/thresholds.json
SAMPLE_INPUT      ?= data/examples/sample_input.json
E2E_OUTPUT        ?= data/examples/e2e_output.json


help:  ## Muestra este mensaje de ayuda
	@echo "Targets disponibles:"
	@echo "  make setup    - Crear entorno conda desde environment.yml"
	@echo "  make train    - Entrenar modelo + calibradores + umbrales"
	@echo "  make predict  - Ejecutar predicción sobre input de ejemplo"
	@echo "  make test     - Ejecutar smoke + golden tests"
	@echo "  make e2e      - Pipeline end-to-end en limpio"
	@echo "  make clean    - Eliminar artefactos generados"
	@echo "  make lint     - Validar esquemas Pydantic contra ejemplos"


# ─── Target 1: setup ─────────────────────────────────────────────────────────

setup:  ## Crear el entorno conda a partir de environment.yml
	@echo ">>> [setup] Creando entorno conda 'posture-risk'..."
	conda env create -f configs/environment.yml || conda env update -f configs/environment.yml --prune
	@echo ">>> [setup] Entorno listo. Actívelo con: conda activate posture-risk"
	@echo ">>> [setup] Registrando kernel Jupyter..."
	conda run -n posture-risk python -m ipykernel install --user --name posture-risk \
		--display-name "Python (posture-risk)"
	@echo ">>> [setup] Verificando estructura de directorios..."
	@$(MKDIR) models 2>/dev/null || true
	@$(MKDIR) logs 2>/dev/null || true
	@$(MKDIR) reports 2>/dev/null || true
	@$(MKDIR) data/examples 2>/dev/null || true
	@echo ">>> [setup] OK"


# ─── Target 2: train ─────────────────────────────────────────────────────────

train:  ## Entrenar modelo actual + calibradores + umbrales
	@echo ">>> [train] Entrenando pipeline completo..."
	$(PYTHON) -m scripts.train_and_persist \
		--config configs/default.yaml \
		--output-dir models/
	@echo ">>> [train] Artefactos generados en models/"


# ─── Target 3: predict ───────────────────────────────────────────────────────

predict:  ## Ejecutar predicción sobre input de ejemplo
	@echo ">>> [predict] Ejecutando predicción..."
	$(PYTHON) -m posture_risk.predict \
		--in $(SAMPLE_INPUT) \
		--out $(E2E_OUTPUT) \
		--model $(MODEL_PATH) \
		--calibrators $(CALIBRATORS_PATH) \
		--thresholds $(THRESHOLDS_PATH)
	@echo ">>> [predict] Salida en $(E2E_OUTPUT)"


# ─── Target 4: test ──────────────────────────────────────────────────────────

test:  ## Ejecutar smoke + golden tests
	@echo ">>> [test] Ejecutando smoke tests..."
	$(PYTHON) -m pytest tests/test_smoke.py -v
	@echo ">>> [test] Ejecutando golden tests..."
	$(PYTHON) -m pytest tests/test_golden.py -v
	@echo ">>> [test] OK"

golden-regenerate:  ## Regenerar el archivo dorado (solo tras cambio intencional del modelo)
	@echo ">>> [golden] Regenerando archivo dorado..."
	$(PYTHON) -m tests.generate_golden
	@echo ">>> [golden] Archivo dorado actualizado. NO OLVIDE hacer git commit."


# ─── Target 5: e2e ───────────────────────────────────────────────────────────

e2e:  ## End-to-end en limpio (setup verificado → train → predict → test)
	@echo "============================================================"
	@echo ">>> [e2e] Pipeline End-to-End en limpio"
	@echo "============================================================"
	@echo ">>> [e2e] 1/5 Verificando entorno..."
	@$(PYTHON) -c "import numpy, pandas, sklearn, xgboost, pydantic; print('  Todas las dependencias OK')"
	@echo ">>> [e2e] 2/5 Verificando artefactos del modelo..."
	@test -f $(MODEL_PATH) || (echo "ERROR: $(MODEL_PATH) no existe. Ejecute 'make train' primero." && exit 1)
	@test -f $(CALIBRATORS_PATH) || (echo "ERROR: $(CALIBRATORS_PATH) no existe." && exit 1)
	@test -f $(THRESHOLDS_PATH) || (echo "ERROR: $(THRESHOLDS_PATH) no existe." && exit 1)
	@echo "  Artefactos verificados"
	@echo ">>> [e2e] 3/5 Ejecutando predicción sobre sample..."
	$(PYTHON) -m posture_risk.predict \
		--in $(SAMPLE_INPUT) \
		--out $(E2E_OUTPUT) \
		--model $(MODEL_PATH) \
		--calibrators $(CALIBRATORS_PATH) \
		--thresholds $(THRESHOLDS_PATH)
	@echo ">>> [e2e] 4/5 Verificando salida..."
	$(PYTHON) scripts/verify_e2e_output.py $(E2E_OUTPUT)
	@echo ">>> [e2e] 5/5 Ejecutando tests..."
	$(PYTHON) -m pytest tests/test_smoke.py tests/test_golden.py -q
	@echo "============================================================"
	@echo ">>> [e2e] PIPELINE COMPLETADO EXITOSAMENTE"
	@echo "============================================================"


# ─── Utilidades ──────────────────────────────────────────────────────────────

lint:  ## Validar ejemplos contra los esquemas Pydantic
	@echo ">>> [lint] Validando examples.json contra esquemas..."
	$(PYTHON) scripts/validate_examples.py contracts/examples.json
	@echo ">>> [lint] OK"

clean:  ## Eliminar artefactos y logs
	@echo ">>> [clean] Eliminando artefactos generados..."
	$(RM) logs/inference_metrics.csv 2>/dev/null || true
	$(RM) data/examples/e2e_output.json 2>/dev/null || true
	@echo ">>> [clean] OK (los modelos entrenados NO se eliminan)"


all: setup train test e2e  ## Setup completo + train + tests + e2e
