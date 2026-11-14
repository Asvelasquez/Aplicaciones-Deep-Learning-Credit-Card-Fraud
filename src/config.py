"""Carga la configuracion del proyecto desde config/config.yaml."""
from pathlib import Path
import yaml

RAIZ = Path(__file__).resolve().parents[1]


def cargar(ruta: str | Path = None) -> dict:
    """Devuelve la configuracion como diccionario."""
    ruta = Path(ruta) if ruta else RAIZ / "config" / "config.yaml"
    with open(ruta, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ruta_absoluta(relativa: str) -> Path:
    """Convierte una ruta del config en ruta absoluta del proyecto."""
    return RAIZ / relativa
