# Detección de fraude en transacciones con tarjeta de crédito

Proyecto de la Especialización en Inteligencia Artificial — Universidad de
Cundinamarca. El objetivo es comparar dos familias de modelos sobre el mismo
problema de datos tabulares: **Gradient Boosting sobre árboles** (XGBoost,
LightGBM) frente a una **red neuronal tabular** (MLP profundo, TabNet).

## El problema

El conjunto contiene 284.807 transacciones europeas de septiembre de 2013, de
las cuales solo 492 son fraudulentas, es decir el 0,17 %. Hay una transacción
fraudulenta por cada 578 legítimas, así que el reto no es acertar mucho sino
encontrar los pocos casos positivos sin llenar de alertas falsas al equipo que
las revisa.

## Estructura del repositorio

```
fraude-tarjeta-dl/
├── config/config.yaml        parámetros del proyecto en un solo lugar
├── data/raw/                 creditcard.csv (no se versiona)
├── data/processed/           particiones y matrices ya transformadas
├── models/                   preprocesador y modelos entrenados
├── notebooks/                exploración de datos
├── reports/                  métricas, figuras y reportes de calidad
└── src/
    ├── config.py             carga de la configuración
    ├── utils.py              logging y semillas
    ├── data/ingest.py        etapa 1 · carga y trazabilidad
    ├── data/validate.py      etapa 1 · reglas de calidad
    ├── data/split.py         etapa 2 · partición temporal
    ├── features/preprocess.py etapa 2 · transformaciones y escalado
    ├── models/train_gbdt.py  etapa 3 · rama A
    ├── models/train_nn.py    etapa 3 · rama B
    └── evaluate.py           etapa 4 · métricas comunes
```

## Instalación

```bash
git clone <url-del-repositorio>
cd fraude-tarjeta-dl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

El archivo `creditcard.csv` se descarga de
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y se coloca
en `data/raw/`. No se versiona porque pesa 144 MB y la licencia es del
repositorio original.

## Cómo correr el pipeline

```bash
python -m src.data.ingest            # 1. carga y registro
python -m src.data.validate          # 2. reglas de calidad
python -m src.data.split             # 3. partición temporal 70 / 15 / 15
python -m src.features.preprocess    # 4. transformación y escalado
python -m src.models.train_gbdt      # 5a. Gradient Boosting
python -m src.models.train_nn        # 5b. red neuronal tabular
python -m src.evaluate --nombre gbdt   # 6. umbral en validación, métricas en prueba
python -m src.evaluate --nombre red
```

Cada etapa se puede correr por separado y deja su resultado en disco, así no
hay que repetir el trabajo anterior cuando se cambia un parámetro.

## Decisiones que conviene conocer

- **La partición es temporal, no aleatoria.** Se ordena por `Time` y se corta
  en bloques, de modo que el modelo se evalúa con transacciones posteriores a
  las que vio. Es más exigente que una partición aleatoria, pero es lo que
  ocurre en producción.
- **El preprocesador se ajusta solo con entrenamiento.** Si se ajustara con
  todo el conjunto, los estadísticos del futuro se filtrarían hacia atrás y
  las métricas quedarían infladas.
- **No se usa accuracy.** Un modelo que diga "ninguna es fraude" acierta el
  99,83 % de las veces y no sirve para nada. La métrica principal es el AUPRC,
  acompañada de recall, precisión y F1.
- **El umbral se ajusta por costo.** Dejar pasar un fraude cuesta mucho más
  que revisar una transacción legítima, así que el punto de corte no se queda
  en 0,5 sino que se busca sobre validación.

## Estado del proyecto

- [x] Avance I — definición del problema, revisión de literatura y diseño
- [x] Avance II — pipeline, recolección y preprocesamiento de datos
- [ ] Avance III — entrenamiento y comparación de los dos modelos
- [ ] Avance IV — análisis de métricas y conclusiones

## Licencia y datos

El código se publica con fines académicos. El conjunto de datos pertenece al
Machine Learning Group de la Université Libre de Bruxelles y se cita según lo
pide su página de Kaggle.
