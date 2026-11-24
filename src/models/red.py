"""Red neuronal tabular en PyTorch: definicion y entrenamiento.

Es el mismo codigo que se ejecuto en el notebook de Colab de los Avances III
y IV, separado en un modulo para poder reutilizarlo desde train_nn.py.

Contenido:
    FocalLoss            perdida focal (Lin et al., 2017)
    EmbeddingPeriodico   embeddings numericos periodicos (Gorishniy et al., 2022)
    RedTabular           bloques Densa + BatchNorm + ReLU + Dropout
    entrenar_red         AdamW, parada temprana por perdida o por AUPRC
    predecir_red         probabilidades por lotes
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


class FocalLoss(nn.Module):
    """Le quita peso a los ejemplos faciles para concentrar el aprendizaje en los fraudes."""

    def __init__(self, alfa: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alfa, self.gamma = alfa, gamma

    def forward(self, logits, y):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p_t = torch.exp(-bce)                       # probabilidad asignada a la clase correcta
        alfa_t = self.alfa * y + (1 - self.alfa) * (1 - y)
        return (alfa_t * (1 - p_t) ** self.gamma * bce).mean()


class EmbeddingPeriodico(nn.Module):
    """Convierte cada variable en senos y cosenos con frecuencias aprendidas."""

    def __init__(self, n_variables: int, k: int = 8, d: int = 16, sigma: float = 0.1):
        super().__init__()
        self.frecuencias = nn.Parameter(torch.randn(n_variables, k) * sigma)
        limite = 1 / math.sqrt(2 * k)
        self.peso = nn.Parameter(torch.empty(n_variables, 2 * k, d).uniform_(-limite, limite))
        self.sesgo = nn.Parameter(torch.zeros(n_variables, d))
        self.n_salida = n_variables * d

    def forward(self, x):
        v = 2 * math.pi * self.frecuencias.unsqueeze(0) * x.unsqueeze(-1)   # (lote, variables, k)
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)                 # (lote, variables, 2k)
        v = torch.einsum("bfi,fid->bfd", v, self.peso) + self.sesgo         # (lote, variables, d)
        return torch.relu(v).flatten(1)


class RedTabular(nn.Module):
    """Perceptron multicapa; opcionalmente con embeddings periodicos a la entrada."""

    def __init__(self, n_entradas, capas=(128, 64, 32), dropout=(0.3, 0.3, 0.2), embedding=None):
        super().__init__()
        self.embedding = embedding
        anterior = embedding.n_salida if embedding is not None else n_entradas
        bloques = []
        for unidades, p in zip(capas, dropout):
            bloques += [nn.Linear(anterior, unidades), nn.BatchNorm1d(unidades), nn.ReLU(), nn.Dropout(p)]
            anterior = unidades
        bloques.append(nn.Linear(anterior, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        return self.red(x).squeeze(1)


def construir(cfg: dict, n_variables: int, razon: float, dispositivo: str):
    """Devuelve el modelo y la funcion de perdida a partir de la configuracion."""
    emb = None
    if cfg.get("embedding"):
        emb = EmbeddingPeriodico(n_variables, cfg.get("emb_k", 8), cfg.get("emb_d", 16), cfg["emb_sigma"])
    modelo = RedTabular(n_variables, cfg["capas"], cfg["dropout"], emb).to(dispositivo)
    if cfg.get("perdida", "focal") == "focal":
        perdida = FocalLoss(cfg["alfa"], cfg["gamma"])
    else:
        perdida = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(razon ** cfg["exp_peso"], device=dispositivo))
    return modelo, perdida


def logits_por_lotes(modelo, X, lote: int = 65536):
    return torch.cat([modelo(X[i:i + lote]) for i in range(0, len(X), lote)])


def predecir_red(modelo, X) -> np.ndarray:
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(logits_por_lotes(modelo, X)).float().cpu().numpy()


def entrenar_red(cfg, X_tr, y_tr, X_va, y_va, razon, dispositivo, semilla=42, trial=None):
    """Entrena la red y devuelve el mejor estado segun el criterio de parada.

    X_tr, X_va: tensores float32 en el dispositivo; y_tr: tensor float32;
    y_va: arreglo numpy con las etiquetas de validacion.
    """
    torch.manual_seed(semilla); np.random.seed(semilla)
    modelo, perdida = construir(cfg, X_tr.shape[1], razon, dispositivo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=dispositivo)
    n, bs = len(X_tr), cfg["batch_size"]
    hist = {"train_loss": [], "val_loss": [], "val_auprc": [], "segundos": []}
    mejor, mejor_estado, mejor_epoca, espera = -np.inf, None, 0, 0

    for epoca in range(cfg["epocas"]):
        t0 = time.time()
        modelo.train()
        perm = torch.randperm(n, device=dispositivo)
        acumulado, pasos = 0.0, 0
        for i in range(0, n - bs + 1, bs):               # se descarta el ultimo lote incompleto
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            l = perdida(modelo(X_tr[idx]), y_tr[idx])
            l.backward()
            opt.step()
            acumulado += l.item(); pasos += 1

        modelo.eval()
        with torch.no_grad():
            logits_va = logits_por_lotes(modelo, X_va)
            l_va = perdida(logits_va, y_va_t).item()
        probs_va = torch.sigmoid(logits_va).float().cpu().numpy()
        if not np.isfinite(l_va) or not np.isfinite(probs_va).all():
            # Divergencia: en Optuna se descarta el trial; fuera de el se conserva el mejor estado.
            if trial is not None:
                import optuna
                raise optuna.TrialPruned()
            if mejor_estado is None:
                raise RuntimeError("La red divergio en la primera epoca")
            break
        auprc_va = float(average_precision_score(y_va, probs_va))
        hist["train_loss"].append(acumulado / pasos)
        hist["val_loss"].append(l_va)
        hist["val_auprc"].append(auprc_va)
        hist["segundos"].append(time.time() - t0)

        puntaje = auprc_va if cfg.get("parada", "val_auprc") == "val_auprc" else -l_va
        if puntaje > mejor:
            mejor, mejor_epoca, espera = puntaje, epoca, 0
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
        if trial is not None:
            trial.report(auprc_va, epoca)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if espera >= cfg["paciencia"]:
            break

    modelo.load_state_dict(mejor_estado)
    hist["mejor_epoca"] = mejor_epoca + 1
    hist["parametros"] = int(sum(p.numel() for p in modelo.parameters()))
    return modelo, hist
