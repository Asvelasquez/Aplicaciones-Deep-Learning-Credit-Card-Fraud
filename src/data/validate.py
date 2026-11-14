"""Etapa 1 del pipeline: reglas de calidad que los datos deben cumplir.

Cada regla devuelve True o False. Si una regla critica falla, el pipeline
se detiene; si falla una regla de aviso, solo queda registrada.

Uso:
    python -m src.data.validate
"""
import json
import pandas as pd

from src.config import cargar, ruta_absoluta
from src.data.ingest import cargar_datos
from src.utils import obtener_logger

log = obtener_logger("validate")


def validar(df: pd.DataFrame, cfg: dict) -> dict:
    """Aplica las reglas de calidad y devuelve el reporte."""
    objetivo = cfg["datos"]["columna_objetivo"]
    esperadas = ([cfg["datos"]["columna_tiempo"]]
                 + cfg["datos"]["columnas_pca"]
                 + [cfg["datos"]["columna_monto"], objetivo])

    faltantes = [c for c in esperadas if c not in df.columns]
    reporte = {
        "criticas": {
            "esquema_completo": len(faltantes) == 0,
            "sin_nulos": int(df.isnull().sum().sum()) == 0,
            "objetivo_binario": sorted(df[objetivo].unique().tolist()) == [0, 1],
            "hay_positivos": int(df[objetivo].sum()) > 0,
        },
        "avisos": {
            "duplicados": int(df.duplicated().sum()),
            "montos_en_cero": int((df[cfg["datos"]["columna_monto"]] == 0).sum()),
            "montos_negativos": int((df[cfg["datos"]["columna_monto"]] < 0).sum()),
        },
        "columnas_faltantes": faltantes,
    }

    for nombre, paso in reporte["criticas"].items():
        log.info("Regla critica %-20s -> %s", nombre, "OK" if paso else "FALLA")
    for nombre, valor in reporte["avisos"].items():
        log.info("Aviso         %-20s -> %s", nombre, valor)

    if not all(reporte["criticas"].values()):
        raise ValueError(f"Validacion fallida: {reporte['criticas']}")
    return reporte


def main() -> dict:
    cfg = cargar()
    df = cargar_datos(ruta_absoluta(cfg["rutas"]["datos_crudos"]))
    reporte = validar(df, cfg)

    destino = ruta_absoluta(cfg["rutas"]["reportes"]) / "validacion.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(reporte, indent=2), encoding="utf-8")
    log.info("Reporte de validacion guardado en %s", destino)
    return reporte


if __name__ == "__main__":
    main()
