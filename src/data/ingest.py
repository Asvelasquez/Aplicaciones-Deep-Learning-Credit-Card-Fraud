"""Etapa 1 del pipeline: carga el archivo crudo y deja registro de lo que entro.

Uso:
    python -m src.data.ingest
"""
from pathlib import Path
import json
import pandas as pd

from src.config import cargar, ruta_absoluta
from src.utils import obtener_logger

log = obtener_logger("ingest")


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV crudo y devuelve el DataFrame."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro {ruta}. Descargue creditcard.csv de Kaggle "
            "y colóquelo en data/raw/."
        )
    df = pd.read_csv(ruta)
    log.info("Cargadas %s filas y %s columnas", f"{len(df):,}", df.shape[1])
    return df


def resumen_ingesta(df: pd.DataFrame, objetivo: str) -> dict:
    """Resumen basico que queda registrado en cada corrida."""
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "positivos": int(df[objetivo].sum()),
        "tasa_positivos_pct": round(100 * float(df[objetivo].mean()), 4),
        "memoria_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }


def main() -> pd.DataFrame:
    cfg = cargar()
    df = cargar_datos(ruta_absoluta(cfg["rutas"]["datos_crudos"]))
    resumen = resumen_ingesta(df, cfg["datos"]["columna_objetivo"])

    destino = ruta_absoluta(cfg["rutas"]["reportes"]) / "ingesta.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    log.info("Resumen de ingesta guardado en %s", destino)
    return df


if __name__ == "__main__":
    main()
