# Yahoo Finance Pipeline

Este proyecto contiene utilidades para seleccionar acciones, procesar datos,
entrenar modelos de predicción y evaluar resultados.

Las partes principales incluyen:

1. **Selección de acciones**: se eligen diez valores basados en volumen,
   estabilidad y rendimiento de los últimos seis meses.
2. **Procesamiento**: descarga de información desde Yahoo Finance y cálculo de
   indicadores técnicos.
3. **Entrenamiento**: se entrenan modelos diarios y semanales y se guardan en
   la carpeta `models`.
4. **Predicción y evaluación**: se aplican los modelos guardados y se registran
   las métricas de precisión y detección de _drift_.
