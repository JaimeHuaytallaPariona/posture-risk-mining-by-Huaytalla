# Tabla comparativa de experimentos A/B

**Métrica principal (★):** F1-Macro — insensible al desbalance.
**Métricas secundarias:** Accuracy, AUC-OvR, AP-Macro.
**Validación:** LOSO (Leave-One-Subject-Out, n=9 folds).

| Modelo                           | F1-Macro ★      | Accuracy        | AUC-OvR         | AP-Macro        |   Tiempo (s) |
|:---------------------------------|:----------------|:----------------|:----------------|:----------------|-------------:|
| Baseline RF (temporal+espectral) | 0.7942 ± 0.2003 | 0.8325 ± 0.1088 | 0.9661 ± 0.0079 | 0.8697 ± 0.1903 |        858.4 |
| Var A — Solo features temporales | 0.7955 ± 0.2004 | 0.8348 ± 0.1077 | 0.9664 ± 0.0086 | 0.8703 ± 0.1906 |        600.4 |
| Var B — SVM Lineal               | 0.7599 ± 0.2037 | 0.7918 ± 0.1428 | 0.9487 ± 0.0117 | 0.8378 ± 0.1801 |       3953.4 |
