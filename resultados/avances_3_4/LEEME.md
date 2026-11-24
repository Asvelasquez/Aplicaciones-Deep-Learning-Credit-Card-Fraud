# Resultados de los Avances III y IV

Salida de `notebooks/avances_3_4_colab.ipynb`, ejecutado en Google Colab con GPU Tesla T4
(Python 3.13, PyTorch 2.14, XGBoost 3.4, LightGBM 4.6, Optuna 5.0, scikit-learn 1.6). La corrida completa tomó 11,6 minutos.

| Archivo | Contenido |
|---|---|
| `tabla_metricas.csv` | Resumen de métricas de todos los modelos en validación y prueba |
| `resultados.json` | Métricas completas, intervalos bootstrap, historiales de entrenamiento, comparaciones pareadas, importancia de variables y análisis de errores |
| `predicciones.npz` | Probabilidades de cada modelo en validación y prueba (`clave__val`, `clave__test`) |
| `etiquetas.npz` | Etiquetas de validación y prueba, montos y hora de las transacciones de prueba |
| `shap_muestra.npz` | Valores SHAP de XGBoost v1 sobre 3.000 transacciones de prueba más todos los fraudes |
| `optuna_xgb.csv` | Los 30 trials de la búsqueda de hiperparámetros de XGBoost |
| `modelos/` | XGBoost v2, Red MLP v2 y Red MLP v2 con embeddings (semilla 42) |
| `figuras/` | Figuras del informe, generadas con `notebooks/figuras_informe.py` |
| `v1_resumen.png`, `v2_resumen.png` | Gráficas generadas por el propio notebook |

En la primera ejecución la búsqueda de hiperparámetros de la red con Optuna se detuvo porque un trial divergió (valores NaN).
El notebook ya incluye la corrección: los trials divergentes se descartan sin detener el estudio.
