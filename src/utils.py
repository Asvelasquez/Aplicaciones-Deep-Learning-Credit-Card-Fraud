"""Utilidades compartidas: registro de eventos y semillas."""
import logging
import random
import numpy as np


def obtener_logger(nombre: str) -> logging.Logger:
    """Logger con formato uniforme para todos los scripts."""
    logger = logging.getLogger(nombre)
    if not logger.handlers:
        manejador = logging.StreamHandler()
        manejador.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(manejador)
        logger.setLevel(logging.INFO)
    return logger


def fijar_semilla(semilla: int = 42) -> None:
    """Fija la semilla para que los resultados se puedan repetir."""
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch
        torch.manual_seed(semilla)
    except ImportError:
        pass
