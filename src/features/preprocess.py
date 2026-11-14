"""Etapa 2 del pipeline: transforma las variables y guarda el preprocesador.

Punto clave: el preprocesador se ajusta solo con el conjunto de
entrenamiento y despues se aplica a validacion y prueba. Si se ajustara con
todo el conjunto, los estadisticos de los datos futuros se filtrarian al
entrenamiento y las metricas quedarian infladas.

Uso:
    python -m src.features.preprocess
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from src.config import cargar, ruta_absoluta
from src.utils import obtener_logger

log = obtener_logger("preprocess")


def derivar_variables(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Agrega la hora del dia y el logaritmo del monto."""
    df = df.copy()
    col_monto = cfg["datos"]["columna_monto"]
    col_tiempo = cfg["datos"]["columna_tiempo"]

    if cfg["preprocesamiento"]["hora_del_dia"]:
        hora = (df[col_tiempo] // 3600) % 24
        # El reloj es circular: las 23 y las 0 estan juntas, no lejos.
        df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    if cfg["preprocesamiento"]["log_monto"]:
        df["monto_log"] = np.log1p(df[col_monto])

    return df.drop(columns=[col_tiempo, col_monto])


def construir_preprocesador(columnas: list, cfg: dict) -> ColumnTransformer:
    """Arma el transformador que escala todas las variables numericas."""
    escalador = (RobustScaler() if cfg["preprocesamiento"]["escalador"] == "robust"
                 else StandardScaler())
    return ColumnTransformer(
        transformers=[("numericas", Pipeline([("escalar", escalador)]), columnas)],
        remainder="drop",
    )


def main():
    cfg = cargar()
    objetivo = cfg["datos"]["columna_objetivo"]
    carpeta = ruta_absoluta(cfg["rutas"]["datos_procesados"])

    partes = {n: pd.read_csv(carpeta / f"{n}.csv") for n in ["train", "val", "test"]}
    partes = {n: derivar_variables(d, cfg) for n, d in partes.items()}

    columnas = [c for c in partes["train"].columns if c != objetivo]
    preprocesador = construir_preprocesador(columnas, cfg)

    # Se ajusta unicamente con entrenamiento.
    preprocesador.fit(partes["train"][columnas])
    log.info("Preprocesador ajustado con %s variables", len(columnas))

    for nombre, parte in partes.items():
        X = preprocesador.transform(parte[columnas])
        y = parte[objetivo].to_numpy()
        np.savez_compressed(carpeta / f"{nombre}_proc.npz", X=X, y=y)
        log.info("%-5s -> matriz %s guardada", nombre, X.shape)

    destino_modelo = ruta_absoluta(cfg["rutas"]["modelos"])
    destino_modelo.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocesador": preprocesador, "columnas": columnas},
                destino_modelo / "preprocessor.joblib")
    log.info("Preprocesador guardado en %s", destino_modelo / "preprocessor.joblib")


if __name__ == "__main__":
    main()
