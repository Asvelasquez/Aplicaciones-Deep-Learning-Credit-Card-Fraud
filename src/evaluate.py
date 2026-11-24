"""Etapa 4 del pipeline: metricas comunes, intervalos de confianza y comparaciones.

El umbral se busca sobre validacion minimizando el costo de los errores y se
aplica sin cambios al conjunto de prueba. Los intervalos de confianza salen de
un bootstrap estratificado (se remuestrean por separado fraudes y legitimas).

Uso:
    python -m src.evaluate --nombre gbdt
    python -m src.evaluate --nombre red
    python -m src.evaluate --nombre gbdt --contra red     # bootstrap pareado
"""
import argparse
import json

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.config import cargar, ruta_absoluta
from src.utils import obtener_logger

log = obtener_logger("evaluate")


def umbral_por_costo(y, p, costo_fn: float, costo_fp: float) -> float:
    """Recorre todos los umbrales posibles de una vez y devuelve el de menor costo."""
    orden = np.argsort(-p, kind="mergesort")
    ps, ys = p[orden], y[orden]
    vp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    costo = (ys.sum() - vp) * costo_fn + fp * costo_fp
    validos = np.where(np.r_[ps[1:] != ps[:-1], True])[0]      # ultimo indice de cada empate
    return float(ps[validos[np.argmin(costo[validos])]])


def metricas(y, p, umbral: float, costo_fn: float, costo_fp: float) -> dict:
    """Metricas apropiadas para clases muy desbalanceadas."""
    pred = (p >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    orden = np.argsort(-p)
    r = {"umbral": umbral, "auprc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
         "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
         "f1": f1_score(y, pred, zero_division=0), "vp": vp, "fp": fp, "fn": fn, "vn": vn,
         "costo": fn * costo_fn + fp * costo_fp, "precision_top100": y[orden[:100]].mean()}
    return {k: round(float(v), 4) for k, v in r.items()}


def _muestras(y, n, semilla):
    rng = np.random.default_rng(semilla)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_auprc(y, p, n=1000, semilla=42):
    v = [average_precision_score(y[i], p[i]) for i in _muestras(y, n, semilla)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def bootstrap_pareado(y, pa, pb, n=1000, semilla=42):
    """Diferencia de AUPRC entre dos modelos sobre las mismas muestras."""
    d = np.array([average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i])
                  for i in _muestras(y, n, semilla)])
    return {"diferencia_media": round(float(d.mean()), 4),
            "ic95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "prob_a_mejor": round(float((d > 0).mean()), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nombre", required=True, help="gbdt, red, red_emb o ensamble")
    parser.add_argument("--contra", default=None, help="segundo modelo para el bootstrap pareado")
    args = parser.parse_args()

    cfg = cargar()
    cfn, cfp = cfg["evaluacion"]["costo_falso_negativo"], cfg["evaluacion"]["costo_falso_positivo"]
    carpeta = ruta_absoluta(cfg["rutas"]["datos_procesados"])
    modelos = ruta_absoluta(cfg["rutas"]["modelos"])
    y_val = np.load(carpeta / "val_proc.npz")["y"]
    y_test = np.load(carpeta / "test_proc.npz")["y"]
    p_val = np.load(modelos / f"probs_{args.nombre}_val.npy")
    p_test = np.load(modelos / f"probs_{args.nombre}_test.npy")

    umbral = umbral_por_costo(y_val, p_val, cfn, cfp)
    resultado = metricas(y_test, p_test, umbral, cfn, cfp)
    resultado["auprc_ic95"] = bootstrap_auprc(y_test, p_test)
    if args.contra:
        p_otro = np.load(modelos / f"probs_{args.contra}_test.npy")
        resultado[f"pareado_vs_{args.contra}"] = bootstrap_pareado(y_test, p_test, p_otro)
    log.info("%s -> %s", args.nombre, json.dumps(resultado))

    destino = ruta_absoluta(cfg["rutas"]["reportes"]) / f"metricas_{args.nombre}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
