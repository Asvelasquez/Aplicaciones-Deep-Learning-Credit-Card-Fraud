"""Etapa 3, rama A: Gradient Boosting sobre arboles.

Se usa XGBoost si esta instalado; si no, se cae en el HistGradientBoosting
de scikit-learn, que sigue el mismo principio y sirve de referencia.

Uso:
    python -m src.models.train_gbdt
"""
import joblib
import numpy as np

from src.config import cargar, ruta_absoluta
from src.utils import fijar_semilla, obtener_logger

log = obtener_logger("train_gbdt")


def construir_modelo(cfg: dict, peso_positivo: float):
    """Devuelve el modelo de boosting disponible en el entorno."""
    p = cfg["entrenamiento"]["gbdt"]
    semilla = cfg["entrenamiento"]["semilla"]
    try:
        from xgboost import XGBClassifier
        log.info("Usando XGBoost con scale_pos_weight=%.1f", peso_positivo)
        return XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            scale_pos_weight=peso_positivo, eval_metric="aucpr",
            tree_method="hist", random_state=semilla, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        log.warning("XGBoost no disponible, se usa HistGradientBoosting")
        return HistGradientBoostingClassifier(
            max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], class_weight="balanced",
            random_state=semilla)


def main():
    cfg = cargar()
    fijar_semilla(cfg["entrenamiento"]["semilla"])
    carpeta = ruta_absoluta(cfg["rutas"]["datos_procesados"])

    entrena = np.load(carpeta / "train_proc.npz")
    valida = np.load(carpeta / "val_proc.npz")
    prueba = np.load(carpeta / "test_proc.npz")
    X, y = entrena["X"], entrena["y"]

    # El desbalance se compensa pesando la clase minoritaria.
    peso = float((y == 0).sum() / max((y == 1).sum(), 1))
    modelo = construir_modelo(cfg, peso)
    modelo.fit(X, y)
    log.info("Entrenamiento terminado con %s transacciones", f"{len(y):,}")

    destino = ruta_absoluta(cfg["rutas"]["modelos"])
    joblib.dump(modelo, destino / "gbdt.joblib")
    np.save(destino / "probs_gbdt_val.npy", modelo.predict_proba(valida["X"])[:, 1])
    np.save(destino / "probs_gbdt_test.npy", modelo.predict_proba(prueba["X"])[:, 1])
    log.info("Modelo y probabilidades guardados en %s", destino)


if __name__ == "__main__":
    main()
