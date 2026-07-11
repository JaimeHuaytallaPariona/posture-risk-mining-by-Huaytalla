# Definición formal de SLOs del sistema

**Documento:** Service Level Objectives (SLOs) del sistema de clasificación postural
**Autor:** Msc. Jaime Antonio Huaytalla Pariona
**Asesor:** Ing. Glen Rodríguez
**Sprint:** 6 — Prueba con stakeholder y latencia
**Estado:** Declarados formalmente antes de la medición (sesión 12, slide 5)

---

## 1. Definición conceptual

Un **Service Level Objective (SLO)** es el objetivo medible de calidad del servicio que el sistema debe cumplir. Es un ítem individual medible de un Service Level Agreement (SLA). Los SLOs se declaran **antes** de la medición, no después de ver los resultados, para evitar sesgo de racionalización.

Cada SLO tiene: nombre, umbral, unidad, operador de comparación, justificación técnica y criticidad. Los SLOs críticos bloquean el despliegue si fallan; los no críticos se registran como deuda técnica pero no impiden avanzar.

---

## 2. SLOs adoptados

### 2.1 Latencia p95 por ventana < 200 ms [CRÍTICO]

Las ventanas de entrada son de 200 ms con 50% de solapamiento. Si el p95 excede 200 ms, no es posible sostener el streaming sin acumulación de cola. El p95 (peor caso realista) es más restrictivo que el p50 y garantiza estabilidad temporal.

### 2.2 Latencia p50 por ventana < 100 ms

La mediana bajo 100 ms asegura que la mayoría de predicciones se procesan con margen amplio respecto al tiempo de ventana (200 ms), dejando espacio para picos ocasionales.

### 2.3 Throughput sostenido ≥ 5 predicciones/segundo [CRÍTICO]

Con ventanas de 200 ms sin solapamiento, se requieren al menos 5 predicciones/segundo. Con el solapamiento del 50% del pipeline actual el requerimiento real sería 10 pred/s; se adopta el umbral conservador de 5 pred/s como criterio mínimo de viabilidad.

### 2.4 Tasa de errores de inferencia < 1% [CRÍTICO]

En un sistema ergonómico de seguridad ocupacional, errores frecuentes de inferencia (excepciones, NaN, timeouts) dejarían al operador sin alertas cuando fueran necesarias, degradando directamente la seguridad.

### 2.5 F1-Macro mínimo ≥ 0.80 [CRÍTICO]

El F1-Macro del sistema en validación LOSO no debe caer por debajo del 0.80 absoluto. Además, debe superar al baseline RF (0.7942) en al menos +2%.

---

## 3. Presupuesto de latencia (slide 5)

La latencia total del sistema es un presupuesto distribuido entre etapas:

```
Latencia total = Preprocesamiento + Escalado + Inferencia XGBoost +
                 Calibración isotónica + Aplicación de umbrales
```

En despliegue con red (no medido en este sprint):
```
Latencia extremo-a-extremo = Adquisición IMU + Transmisión + Procesamiento +
                             Postproceso + Emisión de alerta
```

En esta iteración se mide únicamente el segmento de procesamiento. La adquisición IMU y la emisión de alerta se asumen componentes de hardware con latencia acotada.

---

## 4. Nota metodológica sobre las mediciones

Las latencias se miden en la **PC de desarrollo del investigador**, no en el Jetson Orin Nano objetivo del despliegue. Las cifras son orientativas. La medición en el Jetson real está planeada para cuando se disponga del hardware.

**Implicación:** las cifras en CPU de desarrollo pueden diferir del Jetson Orin Nano en dos direcciones:
- En CPU la latencia puede ser **menor** si la PC de desarrollo es más rápida
- En Jetson la latencia puede ser **menor** si se usa aceleración por GPU (CUDA/TensorRT)

El cumplimiento de los SLOs en esta iteración es **necesario pero no suficiente** para validar el despliegue final; se requiere medición confirmatoria en el hardware objetivo.

---

## 5. Criterios de compliance

- **Sistema APTO para despliegue** ⟺ todos los SLOs críticos pasan
- **Sistema NO APTO** ⟺ al menos un SLO crítico falla
- Un SLO no crítico que falla no bloquea el despliegue pero se documenta como deuda técnica

---

## 6. Evaluación

Los resultados se guardan automáticamente en:
- `reports/sprint6_slo_report.json` — reporte completo en JSON
- `reports/sprint6_summary.json` — resumen consolidado

El notebook `notebooks/08_stakeholder_latency.ipynb` imprime el veredicto de compliance al final.

---

*Documento generado antes de la ejecución del notebook 08 para garantizar que los umbrales no dependan de los valores medidos.*
