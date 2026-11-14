"""Etapa 3, rama B: red neuronal tabular en PyTorch.

Es un perceptron multicapa con normalizacion por lotes y dropout. El
desbalance se maneja con focal loss, que le quita peso a los ejemplos
faciles y obliga a la red a concentrarse en los dificiles, que son los
fraudes.

Uso:
    python -m src.models.train_nn
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import cargar, ruta_absoluta
from src.utils import fijar_semilla, obtener_logger

log = obtener_logger("train_nn")
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class RedTabular(nn.Module):
    """MLP con un bloque densa + BatchNorm + ReLU + Dropout por capa."""

    def __init__(self, n_entradas: int, capas: list, dropout: list):
        super().__init__()
        bloques, anterior = [], n_entradas
        for unidades, p in zip(capas, dropout):
            bloques += [nn.Linear(anterior, unidades), nn.BatchNorm1d(unidades),
                        nn.ReLU(), nn.Dropout(p)]
            anterior = unidades
        bloques.append(nn.Linear(anterior, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x).squeeze(1)


class FocalLoss(nn.Module):
    """Perdida focal: penaliza mas los casos que el modelo falla."""

    def __init__(self, alfa: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alfa, self.gamma = alfa, gamma

    def forward(self, logits, objetivo):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, objetivo, reduction="none")
        prob = torch.exp(-bce)
        peso = self.alfa * objetivo + (1 - self.alfa) * (1 - objetivo)
        return (peso * (1 - prob) ** self.gamma * bce).mean()


def cargar_particion(carpeta, nombre):
    datos = np.load(carpeta / f"{nombre}_proc.npz")
    return (torch.tensor(datos["X"], dtype=torch.float32),
            torch.tensor(datos["y"], dtype=torch.float32))


def main():
    cfg = cargar()
    fijar_semilla(cfg["entrenamiento"]["semilla"])
    p = cfg["entrenamiento"]["red_neuronal"]
    carpeta = ruta_absoluta(cfg["rutas"]["datos_procesados"])

    X_tr, y_tr = cargar_particion(carpeta, "train")
    X_va, y_va = cargar_particion(carpeta, "val")
    X_te, _ = cargar_particion(carpeta, "test")

    cargador = DataLoader(TensorDataset(X_tr, y_tr), batch_size=p["batch_size"],
                          shuffle=True, drop_last=True)
    modelo = RedTabular(X_tr.shape[1], p["capas"], p["dropout"]).to(DISPOSITIVO)
    criterio = FocalLoss()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=p["learning_rate"])

    mejor, espera = float("inf"), 0
    for epoca in range(p["epocas"]):
        modelo.train()
        for xb, yb in cargador:
            xb, yb = xb.to(DISPOSITIVO), yb.to(DISPOSITIVO)
            optimizador.zero_grad()
            perdida = criterio(modelo(xb), yb)
            perdida.backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            perdida_val = criterio(modelo(X_va.to(DISPOSITIVO)),
                                   y_va.to(DISPOSITIVO)).item()
        log.info("Epoca %02d | perdida validacion %.5f", epoca + 1, perdida_val)

        # Parada temprana: si la validacion deja de mejorar, no se sigue.
        if perdida_val < mejor:
            mejor, espera = perdida_val, 0
            torch.save(modelo.state_dict(),
                       ruta_absoluta(cfg["rutas"]["modelos"]) / "red_tabular.pt")
        else:
            espera += 1
            if espera >= p["paciencia"]:
                log.info("Parada temprana en la epoca %s", epoca + 1)
                break

    destino = ruta_absoluta(cfg["rutas"]["modelos"])
    modelo.load_state_dict(torch.load(destino / "red_tabular.pt"))
    modelo.eval()
    with torch.no_grad():
        for nombre, X in [("val", X_va), ("test", X_te)]:
            probs = torch.sigmoid(modelo(X.to(DISPOSITIVO))).cpu().numpy()
            np.save(destino / f"probs_red_{nombre}.npy", probs)
    log.info("Probabilidades de validacion y prueba guardadas en %s", destino)


if __name__ == "__main__":
    main()
