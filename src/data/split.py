"""Etapa 2 del pipeline: separa entrenamiento, validacion y prueba.

La particion por defecto es temporal: se ordena por la columna Time y se
corta en bloques. Asi el modelo se evalua con transacciones posteriores a
las que vio, que es como funcionaria en produccion.

Uso:
    python -m src.data.split
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import cargar, ruta_absoluta
from src.data.ingest import cargar_datos
from src.utils import obtener_logger

log = obtener_logger("split")


def particion_temporal(df: pd.DataFrame, cfg: dict):
    """Corta el conjunto en tres bloques consecutivos en el tiempo."""
    col_tiempo = cfg["datos"]["columna_tiempo"]
    p_tr = cfg["particion"]["proporcion_entrenamiento"]
    p_va = cfg["particion"]["proporcion_validacion"]

    df = df.sort_values(col_tiempo, kind="mergesort").reset_index(drop=True)   # orden estable
    n = len(df)
    corte1, corte2 = int(n * p_tr), int(n * (p_tr + p_va))
    return df.iloc[:corte1], df.iloc[corte1:corte2], df.iloc[corte2:]


def particion_aleatoria(df: pd.DataFrame, cfg: dict):
    """Particion estratificada, util solo como referencia de comparacion."""
    objetivo = cfg["datos"]["columna_objetivo"]
    semilla = cfg["particion"]["semilla"]
    p_va = cfg["particion"]["proporcion_validacion"]
    p_te = 1 - cfg["particion"]["proporcion_entrenamiento"] - p_va

    entrena, resto = train_test_split(
        df, test_size=p_va + p_te, stratify=df[objetivo], random_state=semilla)
    valida, prueba = train_test_split(
        resto, test_size=p_te / (p_va + p_te), stratify=resto[objetivo],
        random_state=semilla)
    return entrena, valida, prueba


def main():
    cfg = cargar()
    df = cargar_datos(ruta_absoluta(cfg["rutas"]["datos_crudos"]))

    if cfg["preprocesamiento"]["eliminar_duplicados"]:
        antes = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        log.info("Duplicados eliminados: %s", antes - len(df))

    if cfg["particion"]["estrategia"] == "temporal":
        partes = particion_temporal(df, cfg)
    else:
        partes = particion_aleatoria(df, cfg)

    destino = ruta_absoluta(cfg["rutas"]["datos_procesados"])
    destino.mkdir(parents=True, exist_ok=True)
    objetivo = cfg["datos"]["columna_objetivo"]

    for nombre, parte in zip(["train", "val", "test"], partes):
        parte.to_csv(destino / f"{nombre}.csv", index=False)
        log.info("%-5s -> %s filas | %s fraudes (%.4f %%)", nombre,
                 f"{len(parte):,}", int(parte[objetivo].sum()),
                 100 * parte[objetivo].mean())
    return partes


if __name__ == "__main__":
    main()
