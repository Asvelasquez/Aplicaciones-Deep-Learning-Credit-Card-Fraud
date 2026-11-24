"""Ensamble de XGBoost y la red: promedio de percentiles.

Los dos modelos entregan probabilidades en escalas distintas (el peso de la
clase las desplaza), por eso se promedia la posicion relativa de cada
transaccion y no la probabilidad cruda.

Uso:
    python -m src.models.ensamble --red red
"""
import argparse

import numpy as np
import pandas as pd

from src.config import cargar, ruta_absoluta
from src.utils import obtener_logger

log = obtener_logger("ensamble")


def percentil(p: np.ndarray) -> np.ndarray:
    return pd.Series(p).rank(pct=True).to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", default="red", help="red o red_emb")
    args = parser.parse_args()
    destino = ruta_absoluta(cargar()["rutas"]["modelos"])
    for parte in ("val", "test"):
        a = np.load(destino / f"probs_gbdt_{parte}.npy")
        b = np.load(destino / f"probs_{args.red}_{parte}.npy")
        np.save(destino / f"probs_ensamble_{parte}.npy", (percentil(a) + percentil(b)) / 2)
    log.info("Ensamble gbdt + %s guardado", args.red)


if __name__ == "__main__":
    main()
