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

## Resultado principal (Avances III y IV)

Sobre el conjunto de prueba (42.559 transacciones, 52 fraudes), con el umbral
elegido en validación:

| Modelo | AUPRC prueba | IC 95 % | Recall | Precisión | Alertas falsas |
|---|---|---|---|---|---|
| Red MLP v2 | 0,772 | 0,661 a 0,870 | 0,769 | 0,476 | 44 |
| XGBoost v2 | 0,764 | 0,656 a 0,861 | 0,750 | 0,506 | 38 |
| Ensamble XGBoost + MLP v2 | 0,773 | 0,663 a 0,871 | 0,750 | 0,582 | 28 |
| Regresión logística (línea base) | 0,692 | 0,551 a 0,837 | 0,788 | 0,318 | 88 |

La diferencia de AUPRC entre XGBoost v2 y la Red MLP v2 es de −0,007, con un
intervalo bootstrap pareado de −0,030 a 0,011: **en este problema la red
neuronal tabular y el Gradient Boosting rinden igual**. El ensamble mantiene el
recall y reduce las alertas falsas. La tabla completa está en
`resultados/avances_3_4/tabla_metricas.csv`.

## Estructura del repositorio

```
fraude-tarjeta-dl/
├── config/config.yaml           parámetros del proyecto (incluye los modelos v2)
├── data/raw/                    creditcard.csv (no se versiona)
├── data/processed/              particiones y matrices ya transformadas
├── models/                      salidas locales de los scripts (no se versiona)
├── notebooks/
│   ├── eda_figuras.py           análisis exploratorio del Avance II
│   ├── avances_3_4_colab.ipynb  implementación, pruebas y rediseño (Avances III y IV)
│   └── figuras_informe.py       figuras del informe a partir de los resultados
├── reports/                     reportes de calidad del pipeline
├── resultados/avances_3_4/      métricas, predicciones, modelos y figuras de Colab
└── src/
    ├── config.py                carga de la configuración
    ├── utils.py                 logging y semillas
    ├── data/ingest.py           etapa 1 · carga y trazabilidad
    ├── data/validate.py         etapa 1 · reglas de calidad
    ├── data/split.py            etapa 2 · partición temporal
    ├── features/preprocess.py   etapa 2 · transformaciones y escalado
    ├── models/train_gbdt.py     etapa 3 · rama A, XGBoost v2 con varias semillas
    ├── models/red.py            etapa 3 · rama B, red tabular en PyTorch
    ├── models/train_nn.py       etapa 3 · rama B, entrenamiento de la Red MLP v2
    ├── models/ensamble.py       etapa 3 · promedio de percentiles
    └── evaluate.py              etapa 4 · métricas, IC bootstrap y comparación pareada
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
en `data/raw/`. No se versiona porque pesa 144 MB, más que el límite de
GitHub, y la licencia es del repositorio original.

## Cómo reproducir los resultados

**Opción 1, notebook de Colab (la que se usó para el informe).** Abrir
`notebooks/avances_3_4_colab.ipynb` en Google Colab con GPU T4 y ejecutar
todas las celdas. Tarda unos 12 minutos y descarga un `.zip` con los mismos
archivos que hay en `resultados/avances_3_4/`.

**Opción 2, scripts del pipeline.**

```bash
python -m src.data.ingest             # 1. carga y registro
python -m src.data.validate           # 2. reglas de calidad
python -m src.data.split              # 3. partición temporal 70 / 15 / 15
python -m src.features.preprocess     # 4. transformación y escalado
python -m src.models.train_gbdt       # 5a. XGBoost v2, tres semillas
python -m src.models.train_nn         # 5b. Red MLP v2, tres semillas (requiere PyTorch)
python -m src.models.ensamble --red red
python -m src.evaluate --nombre gbdt --contra red
python -m src.evaluate --nombre ensamble
```

Cada etapa deja su resultado en disco, así no hay que repetir el trabajo
anterior cuando se cambia un parámetro.

## Decisiones que conviene conocer

- **La partición es temporal, no aleatoria.** Se ordena por `Time` y se corta
  en bloques, de modo que el modelo se evalúa con transacciones posteriores a
  las que vio.
- **El preprocesador se ajusta solo con entrenamiento**, para que la
  información del futuro no se filtre hacia atrás.
- **No se usa accuracy.** La métrica principal es el AUPRC, acompañada de
  recall, precisión, F1 y un costo que pondera cada fraude perdido como 50
  alertas falsas.
- **El umbral, los hiperparámetros y la parada temprana se deciden con
  validación.** El conjunto de prueba solo se usa para medir.
- **La red se detiene por AUPRC y no por pérdida.** En las pruebas iniciales
  la pérdida eligió la época 21 cuando el mejor AUPRC estaba en la época 6.
- **Cada modelo final promedia tres semillas**, y las comparaciones se hacen
  con bootstrap pareado porque con 52 fraudes en prueba las diferencias
  pequeñas son ruido.

## Estado del proyecto · 87,5 %

- [x] Avance I — definición del problema, revisión de literatura y diseño
- [x] Avance II — pipeline, recolección y preprocesamiento de datos
- [x] Avance III — implementación, pruebas iniciales y propuesta de rediseño
- [x] Avance IV — rediseño, comparación estadística y métricas finales
- [ ] Pendiente — repetir la búsqueda de Optuna de la red con la corrección de divergencia
- [ ] Avances V y VI — entregas finales

## Licencia y datos

El código se publica con fines académicos. El conjunto de datos pertenece al
Machine Learning Group de la Université Libre de Bruxelles y se cita según lo
pide su página de Kaggle.
