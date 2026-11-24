"""Inferencia de referencia en NumPy para el modelo desplegado.

Reproduce sin PyTorch ni XGBoost la Red MLP v2 y XGBoost v2 (semilla 42),
con el mismo preprocesamiento del pipeline. Sirve como referencia para
verificar que la version en JavaScript (docs/js/modelo.js) calcula lo mismo.
"""
import json, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from src.deploy.cargar_pt import cargar_state_dict

COLS_V = [f"V{i}" for i in range(1, 29)]

def derivar(df):
    X = df[COLS_V].copy()
    hora = (df["Time"] // 3600) % 24
    X["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    X["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    X["monto_log"] = np.log1p(df["Amount"])
    return X

def particion(df):
    df = df.drop_duplicates().sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(df); c1, c2 = int(n * 0.70), int(n * 0.85)
    return df.iloc[:c1], df.iloc[c1:c2], df.iloc[c2:]

class Red:
    def __init__(self, ruta_pt):
        sd = cargar_state_dict(ruta_pt)
        self.capas = []
        for lin, bn in [(0, 1), (4, 5), (8, 9)]:
            self.capas.append({
                "W": sd[f"red.{lin}.weight"].astype(np.float64), "b": sd[f"red.{lin}.bias"].astype(np.float64),
                "gamma": sd[f"red.{bn}.weight"].astype(np.float64), "beta": sd[f"red.{bn}.bias"].astype(np.float64),
                "media": sd[f"red.{bn}.running_mean"].astype(np.float64), "var": sd[f"red.{bn}.running_var"].astype(np.float64)})
        self.W_out = sd["red.12.weight"].astype(np.float64); self.b_out = sd["red.12.bias"].astype(np.float64)
    def predecir(self, X):
        h = X.astype(np.float64)
        for c in self.capas:
            h = h @ c["W"].T + c["b"]
            h = (h - c["media"]) / np.sqrt(c["var"] + 1e-5) * c["gamma"] + c["beta"]   # BatchNorm en evaluacion
            h = np.maximum(h, 0)                                                          # ReLU (dropout no actua)
        z = (h @ self.W_out.T + self.b_out)[:, 0]
        return 1 / (1 + np.exp(-z))

class Bosque:
    def __init__(self, ruta_json):
        m = json.load(open(ruta_json))
        L = m["learner"]
        n_arboles = int(L["attributes"]["best_iteration"]) + 1
        p0 = float(L["learner_model_param"]["base_score"].strip("[]"))
        self.margen_base = math.log(p0 / (1 - p0))
        self.arboles = [{k: np.array(t[k]) for k in ("left_children", "right_children", "split_indices",
                                                     "split_conditions", "default_left")}
                        for t in L["gradient_booster"]["model"]["trees"][:n_arboles]]
        for t in self.arboles:
            t["split_conditions"] = t["split_conditions"].astype(np.float32)
    def predecir(self, X):
        X = X.astype(np.float32)
        margen = np.full(len(X), self.margen_base)
        for t in self.arboles:
            nodo = np.zeros(len(X), dtype=np.int64)
            activos = t["left_children"][nodo] != -1
            while activos.any():
                n = nodo[activos]
                x = X[np.where(activos)[0], t["split_indices"][n]]
                izq = np.where(np.isnan(x), t["default_left"][n] == 1, x < t["split_conditions"][n])
                nodo[activos] = np.where(izq, t["left_children"][n], t["right_children"][n])
                activos = t["left_children"][nodo] != -1
            margen += t["split_conditions"][nodo]
        return 1 / (1 + np.exp(-margen))


def tabla_percentiles(puntajes):
    """Cuantiles de los puntajes de validacion, mas densos en la cola alta donde estan los fraudes."""
    q = np.concatenate([np.linspace(0, 0.99, 100), np.linspace(0.99, 0.999, 91)[1:], np.linspace(0.999, 1, 101)[1:]])
    v = np.quantile(puntajes, q)
    # valores estrictamente crecientes: para empates se conserva el cuantil mas alto
    xs, ys = [], []
    for valor, cuantil in zip(v, q):
        if xs and valor <= xs[-1]:
            ys[-1] = cuantil
        else:
            xs.append(float(valor)); ys.append(float(cuantil))
    return {"valores": xs, "cuantiles": ys}

def percentil(p, tabla):
    return np.interp(p, tabla["valores"], tabla["cuantiles"])

def puntaje_ensamble(p_red, p_bosque, tabla_red, tabla_bosque):
    return (percentil(p_red, tabla_red) + percentil(p_bosque, tabla_bosque)) / 2

def umbral_por_costo(y, p, costo_fn=1.0, costo_fp=0.02):
    orden = np.argsort(-p, kind="mergesort")
    ps, ys = p[orden], y[orden]
    vp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    costo = (ys.sum() - vp) * costo_fn + fp * costo_fp
    validos = np.where(np.r_[ps[1:] != ps[:-1], True])[0]
    return float(ps[validos[np.argmin(costo[validos])]])
