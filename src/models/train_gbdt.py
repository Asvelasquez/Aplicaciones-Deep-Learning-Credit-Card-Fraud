"""Etapa 3, rama A: XGBoost v2 con varias semillas.

Los hiperparametros vienen de la busqueda con Optuna del Avance IV y estan
en config.yaml. Cada semilla entrena un modelo con parada temprana sobre el
AUPRC de validacion; la prediccion final es el promedio de las semillas.

Si XGBoost no esta instalado se usa HistGradientBoosting de scikit-learn,
que sigue el mismo principio, solo como respaldo para probar el pipeline.

Uso:
    python -m src.models.train_gbdt
"""
import json

import joblib
import numpy as np

from src.config import cargar, ruta_absoluta
from src.utils import fijar_semilla, obtener_logger

log = obtener_logger("train_gbdt")


def construir_modelo(p: dict, razon: float, semilla: int):
    """Devuelve el modelo de boosting disponible en el entorno."""
    try:
        import xgboost as xgb
        try:
            import torch
            dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dispositivo = "cpu"
        return xgb.XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            min_child_weight=p["min_child_weight"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], gamma=p["gamma"], reg_lambda=p["reg_lambda"],
            scale_pos_weight=razon ** p["exp_peso"], objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", device=dispositivo, early_stopping_rounds=p["parada_temprana"],
            random_state=semilla, n_jobs=-1), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        log.warning("XGBoost no disponible, se usa HistGradientBoosting como respaldo")
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1, random_state=semilla), "sklearn"


def main():
    cfg = cargar()
    p = cfg["entrenamiento"]["gbdt"]
    carpeta = ruta_absoluta(cfg["rutas"]["datos_procesados"])
    destino = ruta_absoluta(cfg["rutas"]["modelos"])
    destino.mkdir(parents=True, exist_ok=True)

    tr, va, te = (np.load(carpeta / f"{n}_proc.npz") for n in ("train", "val", "test"))
    razon = float((tr["y"] == 0).sum() / (tr["y"] == 1).sum())

    p_va, p_te, resumen = [], [], []
    for s in cfg["entrenamiento"]["semillas_finales"]:
        fijar_semilla(s)
        modelo, motor = construir_modelo(p, razon, s)
        if motor == "xgboost":
            modelo.fit(tr["X"], tr["y"], eval_set=[(va["X"], va["y"])], verbose=False)
            iteraciones = int(modelo.best_iteration) + 1
        else:
            modelo.fit(tr["X"], tr["y"])
            iteraciones = int(modelo.n_iter_)
        p_va.append(modelo.predict_proba(va["X"])[:, 1]); p_te.append(modelo.predict_proba(te["X"])[:, 1])
        joblib.dump(modelo, destino / f"gbdt_semilla{s}.joblib")
        resumen.append({"semilla": s, "motor": motor, "arboles": iteraciones})
        log.info("Semilla %s (%s) -> %s arboles", s, motor, iteraciones)

    np.save(destino / "probs_gbdt_val.npy", np.mean(p_va, 0))
    np.save(destino / "probs_gbdt_test.npy", np.mean(p_te, 0))
    (destino / "gbdt_semillas.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    log.info("Probabilidades promedio guardadas en %s", destino)


if __name__ == "__main__":
    main()
