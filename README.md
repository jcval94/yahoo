# Yahoo Finance Pipeline

Este proyecto ofrece una serie de scripts para crear un flujo de trabajo completo con datos financieros descargados de Yahoo Finance. La idea es proporcionar un ejemplo sencillo que muestre desde la descarga de datos hasta la evaluación de modelos y la generación de recomendaciones.

Aunque el código es mínimo, sirve como plantilla educativa. Cada paso se
encuentra aislado en un módulo para que puedas estudiarlo y modificarlo según
tus necesidades. No necesitas ser experto en Python; basta con seguir las
instrucciones e ir probando.

El objetivo es que puedas entender todo el recorrido: desde la selección de los
tickers, pasando por la construcción del dataset, hasta el cálculo de métricas y
la creación de un portafolio. También se incluye un ejemplo de notificación para
cerrar el ciclo completo.

## Diagrama general

```mermaid
flowchart LR
    A[Seleccion de acciones] --> B[Extraccion de datos]
    B --> C[Preprocesamiento]
    C --> D[Entrenamiento de modelos]
    D --> E[Prediccion]
    E --> F[Evaluacion]
    F --> G[Optimizacion de portafolio]
    G --> H[Notificacion]
```

Este flujo se ejecuta normalmente de forma programada mediante GitHub Actions.

## Requisitos

* Python 3.11 o superior
* Las dependencias listadas en `requirements.txt`

Instala todo con:

```bash
pip install -r requirements.txt
```

## Configuracion rapida

El archivo `config.yaml` define los ETFs que se van a procesar y otras opciones básicas:

```yaml
etfs:
  - SPY
  - QQQ
start_date: "2015-01-01"
prediction_horizon: 5
```

Cambia este archivo segun tus necesidades.

## Estructura de carpetas

```mermaid
flowchart TB
    subgraph src
      direction TB
      A1[abt] -- build_abt.py --> A2[Generar ABT]
      B1[models] -- rf_model.py --> B2[Modelos ML]
      C1[portfolio] -- backtest.py --> C2[Portafolio]
    end
    src -->|utilidades| utils
    src --> preprocess
    src --> training
    src --> predict
    src --> evaluation
```

La carpeta `src` contiene todas las utilidades. Algunos scripts son simples plantillas para mostrar donde agregar tu logica.

* `abt/` incluye la construcción de la "Analytic Base Table". Aquí se descargan y
  enriquecen los datos diarios.
* `models/` contiene ejemplos de modelos de machine learning listos para usar o
  para que los sustituyas por los tuyos.
* `portfolio/` alberga herramientas para backtesting y optimización de cartera.
* `notify/` muestra cómo enviar un correo o mensaje una vez que tienes nuevos
  resultados.

Además de estas carpetas, existen scripts de selección y predicción en la raíz
del paquete para que puedas ejecutar el flujo sin complicaciones.

## Ejecucion paso a paso

1. **Seleccion de acciones**

   Ejecuta:
   ```bash
   python -m src.selection
   ```
   Obtendras una lista de los tickers mas interesantes según volumen, estabilidad y desempeño. Esta lista se usa como punto de partida para el resto del flujo.

2. **Descarga y preprocesamiento**
   
   ```bash
   python -m src.abt.build_abt
   ```
   Esto baja datos historicos y agrega indicadores tecnicos. Antes de ejecutarlo puedes editar `config.yaml` para cambiar los tickers o el rango de fechas.

3. **Entrenamiento**
   
   ```bash
   python -m src.training
   ```
   Se generan varios modelos de ejemplo y se guardan en `models/`. Puedes revisar esos archivos para entender cómo mejorar la calidad de las predicciones.

4. **Prediccion**
   
   ```bash
   python -m src.predict
   ```
   Se aplican los modelos guardados y se genera un archivo `predictions.csv` en la carpeta `results/`.

5. **Evaluacion**
   
   ```bash
   python -m src.evaluation
   ```
   Compara predicciones con valores reales y emite métricas simples como MAE y R2. Sirve para saber si los modelos deben reentrenarse.

6. **Optimizacion de portafolio**
   
   ```bash
   python -m src.portfolio.optimize
   ```
   Ajusta los pesos segun tus reglas de negocio para construir un portafolio equilibrado.

7. **Notificacion**

   ```bash
   python -m src.notify.notifier --message "Proceso completo"
   ```
   Envía un aviso por correo o chat con los resultados finales. Es un ejemplo que puedes adaptar a tu sistema de mensajería.

## Flujo de entrenamiento y prediccion

```mermaid
sequenceDiagram
    participant Usuario
    participant ABT as "build_abt.py"
    participant Train as "training.py"
    participant Pred as "predict.py"

    Usuario->>ABT: Extraer y enriquecer datos
    ABT->>Train: Entrega datos procesados
    Train->>Pred: Guarda modelos entrenados
    Pred->>Usuario: Devuelve CSV con predicciones
```

## Automatizacion

El repositorio incluye flujos de trabajo en `.github/workflows` que ejecutan el pipeline de manera programada. Revisa `train.yml` para ver un ejemplo de ejecucion semanal.

## Diagrama del pipeline automatizado

El siguiente esquema muestra cómo cada script se encadena cuando se ejecuta desde
GitHub Actions o desde tu propio entorno local:

```mermaid
flowchart TD
    GA[GitHub Actions] --> S[python -m src.selection]
    S --> ABT[python -m src.abt.build_abt]
    ABT --> TR[python -m src.training]
    TR --> PR[python -m src.predict]
    PR --> EV[python -m src.evaluation]
    EV --> OP[python -m src.portfolio.optimize]
    OP --> NT[python -m src.notify.notifier]
```

Cada bloque representa la ejecucion de un modulo. Si prefieres hacerlo
manualmente, puedes ejecutar cada comando en tu terminal siguiendo el orden del
diagrama.

## Contribuciones

Estas utilidades son un punto de partida. Puedes reemplazar las secciones marcadas como "placeholder" con implementaciones mas robustas. Se aceptan mejoras y comentarios.

