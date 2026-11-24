"""Etapa 3, rama B: entrena la Red MLP v2 con varias semillas.

La prediccion final es el promedio de las semillas configuradas. Con
embedding: true en config.yaml se entrena la variante con embeddings
periodicos. Requiere PyTorch; con GPU tarda menos de un minuto.

Uso:
    python -m src.models.train_nn
"""
import json

import numpy as np
import torch

from src.config import cargar, ruta_absoluta
from src.models.red import entrenar_red, predecir_red
from src.utils import obtener_logger

log = obtener_logger("train_nn")
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def tensores(carpeta, nombre):
    datos = np.load(carpeta / f"{nombre}_proc.npz")
    X = torch.tensor(datos["X"], dtype=torch.float32, device=DISPOSITIVO)
    return X, datos["y"]


def main():
    cfg_total = cargar()
    cfg = cfg_total["entrenamiento"]["red_neuronal"]
    semillas = cfg_total["entrenamiento"]["semillas_finales"]
    carpeta = ruta_absoluta(cfg_total["rutas"]["datos_procesados"])
    destino = ruta_absoluta(cfg_total["rutas"]["modelos"])
    destino.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = tensores(carpeta, "train")
    X_va, y_va = tensores(carpeta, "val")
    X_te, _ = tensores(carpeta, "test")
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DISPOSITIVO)
    razon = float((y_tr == 0).sum() / (y_tr == 1).sum())
    nombre = "red_emb" if cfg.get("embedding") else "red"

    p_va, p_te, resumen = [], [], []
    for s in semillas:
        modelo, hist = entrenar_red(cfg, X_tr, y_tr_t, X_va, y_va, razon, DISPOSITIVO, semilla=s)
        p_va.append(predecir_red(modelo, X_va)); p_te.append(predecir_red(modelo, X_te))
        torch.save(modelo.state_dict(), destino / f"{nombre}_semilla{s}.pt")
        resumen.append({"semilla": s, "epocas": len(hist["val_auprc"]), "mejor_epoca": hist["mejor_epoca"],
                        "val_auprc_max": max(hist["val_auprc"])})
        log.info("Semilla %s -> AUPRC validacion %.4f (epoca %s)", s, max(hist["val_auprc"]), hist["mejor_epoca"])

    np.save(destino / f"probs_{nombre}_val.npy", np.mean(p_va, 0))
    np.save(destino / f"probs_{nombre}_test.npy", np.mean(p_te, 0))
    (destino / f"{nombre}_semillas.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    log.info("Probabilidades promedio guardadas en %s", destino)


if __name__ == "__main__":
    main()
