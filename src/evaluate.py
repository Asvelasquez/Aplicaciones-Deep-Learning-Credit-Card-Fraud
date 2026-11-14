"""Etapa 4 del pipeline: metricas comunes para los dos modelos.

Todas las metricas se calculan igual para Gradient Boosting y para la red
neuronal, de manera que la comparacion sea justa.

El umbral se busca sobre validacion y despues se aplica, sin tocarlo, al
conjunto de prueba. Si se buscara sobre prueba, la metrica final estaria
contaminada por la misma informacion que se quiere medir.

Uso:
    python -m src.evaluate --nombre gbdt
    python -m src.evaluate --nombre red
"""
import argparse
import json
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src.config import cargar, ruta_absoluta
from src.utils import obtener_logger

log = obtener_logger("evaluate")


def mejor_umbral(y_real, y_prob, costo_fn: float, costo_fp: float) -> float:
    """Busca el umbral que minimiza el costo total de los errores."""
    precision, recall, umbrales = precision_recall_curve(y_real, y_prob)
    mejor, mejor_costo = 0.5, float("inf")
    for u in umbrales:
        pred = (y_prob >= u).astype(int)
        fn = int(((pred == 0) & (y_real == 1)).sum())
        fp = int(((pred == 1) & (y_real == 0)).sum())
        costo = fn * costo_fn + fp * costo_fp
        if costo < mejor_costo:
            mejor, mejor_costo = float(u), costo
    return mejor


def metricas(y_real, y_prob, umbral: float = 0.5) -> dict:
    """Conjunto de metricas apropiado para clases muy desbalanceadas."""
    pred = (y_prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, pred).ravel()
    return {
        "umbral": round(umbral, 4),
        "auprc": round(float(average_precision_score(y_real, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_real, y_prob)), 4),
        "precision": round(float(precision_score(y_real, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_real, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_real, pred, zero_division=0)), 4),
        "matriz": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nombre", required=True,
                        help="gbdt o red, segun el modelo a evaluar")
    args = parser.parse_args()

    cfg = cargar()
    carpeta = ruta_absoluta(cfg["rutas"]["datos_procesados"])
    modelos = ruta_absoluta(cfg["rutas"]["modelos"])
    y_val = np.load(carpeta / "val_proc.npz")["y"]
    y_test = np.load(carpeta / "test_proc.npz")["y"]
    p_val = np.load(modelos / f"probs_{args.nombre}_val.npy")
    p_test = np.load(modelos / f"probs_{args.nombre}_test.npy")

    umbral = mejor_umbral(y_val, p_val,
                          cfg["evaluacion"]["costo_falso_negativo"],
                          cfg["evaluacion"]["costo_falso_positivo"])
    resultado = metricas(y_test, p_test, umbral)
    log.info("%s -> %s", args.nombre, json.dumps(resultado))

    destino = ruta_absoluta(cfg["rutas"]["reportes"]) / f"metricas_{args.nombre}.json"
    destino.write_text(json.dumps(resultado, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
