# =============================================================================
# make.ps1 - Wrapper PowerShell del Makefile para Windows nativo
#
# Sesión 13, slide 10: 5 targets mínimos (setup, train, predict, test, e2e)
#
# Uso desde PowerShell:
#   .\make.ps1 setup
#   .\make.ps1 train
#   .\make.ps1 predict
#   .\make.ps1 test
#   .\make.ps1 e2e
#   .\make.ps1 help
# =============================================================================

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

# Cargar .env si existe
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+?)\s*=\s*(.*?)\s*$") {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

# Defaults (sobreescribibles vía .env)
$MODEL_PATH       = if ($env:MODEL_PATH)       { $env:MODEL_PATH }       else { "models\best_xgb.pkl" }
$CALIBRATORS_PATH = if ($env:CALIBRATORS_PATH) { $env:CALIBRATORS_PATH } else { "models\calibrators.pkl" }
$THRESHOLDS_PATH  = if ($env:THRESHOLDS_PATH)  { $env:THRESHOLDS_PATH }  else { "models\thresholds.json" }
$SAMPLE_INPUT     = if ($env:SAMPLE_INPUT)     { $env:SAMPLE_INPUT }     else { "data\examples\sample_input.json" }
$E2E_OUTPUT       = if ($env:E2E_OUTPUT)       { $env:E2E_OUTPUT }       else { "data\examples\e2e_output.json" }


function Show-Help {
    Write-Host "Targets disponibles:" -ForegroundColor Cyan
    Write-Host "  .\make.ps1 setup    - Crear entorno conda desde environment.yml"
    Write-Host "  .\make.ps1 train    - Entrenar modelo + calibradores + umbrales"
    Write-Host "  .\make.ps1 predict  - Ejecutar predicción sobre input de ejemplo"
    Write-Host "  .\make.ps1 test     - Ejecutar smoke + golden tests"
    Write-Host "  .\make.ps1 e2e      - Pipeline end-to-end en limpio"
    Write-Host "  .\make.ps1 clean    - Eliminar artefactos generados"
    Write-Host "  .\make.ps1 lint     - Validar esquemas Pydantic contra ejemplos"
    Write-Host "  .\make.ps1 golden-regenerate - Regenerar archivo dorado"
}

function Ensure-Directories {
    New-Item -ItemType Directory -Force -Path "models"       | Out-Null
    New-Item -ItemType Directory -Force -Path "logs"         | Out-Null
    New-Item -ItemType Directory -Force -Path "reports"      | Out-Null
    New-Item -ItemType Directory -Force -Path "data\examples" | Out-Null
}

function Setup {
    Write-Host ">>> [setup] Creando entorno conda 'posture-risk'..." -ForegroundColor Green
    conda env create -f configs\environment.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> [setup] Entorno ya existe. Actualizando..." -ForegroundColor Yellow
        conda env update -f configs\environment.yml --prune
    }
    Write-Host ">>> [setup] Registrando kernel Jupyter..." -ForegroundColor Green
    conda run -n posture-risk python -m ipykernel install --user --name posture-risk `
        --display-name "Python (posture-risk)"
    Ensure-Directories
    Write-Host ">>> [setup] OK. Actívelo con: conda activate posture-risk" -ForegroundColor Green
}

function Train {
    Write-Host ">>> [train] Entrenando pipeline completo..." -ForegroundColor Green
    python -m scripts.train_and_persist --config configs\default.yaml --output-dir models\
    if ($LASTEXITCODE -ne 0) { throw "Train falló" }
    Write-Host ">>> [train] Artefactos generados en models\" -ForegroundColor Green
}

function Predict {
    Write-Host ">>> [predict] Ejecutando predicción..." -ForegroundColor Green
    python -m posture_risk.predict `
        --in $SAMPLE_INPUT --out $E2E_OUTPUT `
        --model $MODEL_PATH --calibrators $CALIBRATORS_PATH --thresholds $THRESHOLDS_PATH
    if ($LASTEXITCODE -ne 0) { throw "Predict falló" }
    Write-Host ">>> [predict] Salida en $E2E_OUTPUT" -ForegroundColor Green
}

function Test {
    Write-Host ">>> [test] Ejecutando smoke tests..." -ForegroundColor Green
    python -m pytest tests\test_smoke.py -v
    if ($LASTEXITCODE -ne 0) { throw "Smoke tests fallaron" }

    Write-Host ">>> [test] Ejecutando golden tests..." -ForegroundColor Green
    python -m pytest tests\test_golden.py -v
    if ($LASTEXITCODE -ne 0) { throw "Golden tests fallaron" }

    Write-Host ">>> [test] OK" -ForegroundColor Green
}

function Golden-Regenerate {
    Write-Host ">>> [golden] Regenerando archivo dorado..." -ForegroundColor Yellow
    python -m tests.generate_golden
    Write-Host ">>> [golden] Archivo dorado actualizado. NO OLVIDE hacer git commit." -ForegroundColor Yellow
}

function E2E {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ">>> [e2e] Pipeline End-to-End en limpio" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    Write-Host ">>> [e2e] 1/5 Verificando entorno..." -ForegroundColor Green
    python -c "import numpy, pandas, sklearn, xgboost, pydantic; print('  Todas las dependencias OK')"
    if ($LASTEXITCODE -ne 0) { throw "Dependencias no verificables" }

    Write-Host ">>> [e2e] 2/5 Verificando artefactos del modelo..." -ForegroundColor Green
    foreach ($p in $MODEL_PATH, $CALIBRATORS_PATH, $THRESHOLDS_PATH) {
        if (-not (Test-Path $p)) {
            Write-Host "ERROR: $p no existe. Ejecute '.\make.ps1 train' primero." -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "  Artefactos verificados" -ForegroundColor Green

    Write-Host ">>> [e2e] 3/5 Ejecutando predicción sobre sample..." -ForegroundColor Green
    python -m posture_risk.predict `
        --in $SAMPLE_INPUT --out $E2E_OUTPUT `
        --model $MODEL_PATH --calibrators $CALIBRATORS_PATH --thresholds $THRESHOLDS_PATH
    if ($LASTEXITCODE -ne 0) { throw "Predicción falló" }

    Write-Host ">>> [e2e] 4/5 Verificando salida..." -ForegroundColor Green
    python scripts\verify_e2e_output.py $E2E_OUTPUT
    if ($LASTEXITCODE -ne 0) { throw "Verificación de salida falló" }

    Write-Host ">>> [e2e] 5/5 Ejecutando tests..." -ForegroundColor Green
    python -m pytest tests\test_smoke.py tests\test_golden.py -q
    if ($LASTEXITCODE -ne 0) { throw "Tests fallaron" }

    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ">>> [e2e] PIPELINE COMPLETADO EXITOSAMENTE" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
}

function Lint {
    Write-Host ">>> [lint] Validando examples.json contra esquemas..." -ForegroundColor Green
    python scripts\validate_examples.py contracts\examples.json
    if ($LASTEXITCODE -ne 0) { throw "Lint falló" }
    Write-Host ">>> [lint] OK" -ForegroundColor Green
}

function Clean {
    Write-Host ">>> [clean] Eliminando artefactos generados..." -ForegroundColor Green
    Remove-Item -Force "logs\inference_metrics.csv" -ErrorAction SilentlyContinue
    Remove-Item -Force "data\examples\e2e_output.json" -ErrorAction SilentlyContinue
    Write-Host ">>> [clean] OK (los modelos entrenados NO se eliminan)" -ForegroundColor Green
}

# ─── Dispatch ────────────────────────────────────────────────────────────────

switch ($Target.ToLower()) {
    "help"    { Show-Help }
    "setup"   { Setup }
    "train"   { Train }
    "predict" { Predict }
    "test"    { Test }
    "e2e"     { E2E }
    "lint"    { Lint }
    "clean"   { Clean }
    "golden-regenerate" { Golden-Regenerate }
    "all"     { Setup; Train; Test; E2E }
    default   {
        Write-Host "Target desconocido: $Target" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
