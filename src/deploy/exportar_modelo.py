"""Exporta el modelo desplegado a archivos JSON que lee la pagina web (docs/modelo/).

Genera:
    preprocesamiento.json   columnas, centro y escala del RobustScaler
    red_mlp.json            pesos de la Red MLP v2 (semilla 42), con BatchNorm
    xgboost.json            arboles de XGBoost v2 (semilla 42) hasta la mejor iteracion
    referencia.json         tablas de percentiles, umbrales, metricas, rangos y ejemplos

Uso:
    python -m src.deploy.exportar_modelo
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.deploy.cargar_pt import cargar_state_dict
from src.deploy.inferencia import (COLS_V, Bosque, Red, derivar, particion, percentil,
                                   puntaje_ensamble, tabla_percentiles, umbral_por_costo)

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "resultados" / "avances_3_4" / "modelos"
DESTINO = RAIZ / "docs" / "modelo"
EJEMPLOS = RAIZ / "docs" / "ejemplos"
r = lambda a, d=7: [round(float(x), d) for x in np.ravel(a)]


def metricas(y, p, u, montos):
    pred = (p >= u).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auprc": round(float(average_precision_score(y, p)), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "recall": round(vp / (vp + fn), 4), "precision": round(vp / max(vp + fp, 1), 4),
            "f1": round(2 * vp / max(2 * vp + fp + fn, 1), 4), "vp": int(vp), "fp": int(fp), "fn": int(fn), "vn": int(vn),
            "monto_detectado_pct": round(100 * montos[(y == 1) & (pred == 1)].sum() / montos[y == 1].sum(), 1)}


def main():
    DESTINO.mkdir(parents=True, exist_ok=True); EJEMPLOS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAIZ / "data" / "raw" / "creditcard.csv")
    tr, va, te = particion(df)
    esc = RobustScaler().fit(derivar(tr))
    columnas = list(derivar(tr).columns)
    Xva = esc.transform(derivar(va)).astype(np.float32); Xte = esc.transform(derivar(te)).astype(np.float32)

    # 1) preprocesamiento
    json.dump({"entrada": ["Time"] + COLS_V + ["Amount"], "columnas_modelo": columnas,
               "centro": r(esc.center_, 9), "escala": r(esc.scale_, 9)},
              open(DESTINO / "preprocesamiento.json", "w"), indent=0)

    # 2) red neuronal
    sd = cargar_state_dict(RES / "mlp_v2_semilla42.pt")
    capas = []
    for lin, bn in [(0, 1), (4, 5), (8, 9)]:
        capas.append({"W": [r(fila) for fila in sd[f"red.{lin}.weight"]], "b": r(sd[f"red.{lin}.bias"]),
                      "gamma": r(sd[f"red.{bn}.weight"]), "beta": r(sd[f"red.{bn}.bias"]),
                      "media": r(sd[f"red.{bn}.running_mean"]), "var": r(sd[f"red.{bn}.running_var"])})
    json.dump({"arquitectura": "31 -> 128 -> 64 -> 32 -> 1, BatchNorm + ReLU", "eps": 1e-5, "capas": capas,
               "salida": {"W": r(sd["red.12.weight"]), "b": r(sd["red.12.bias"])}},
              open(DESTINO / "red_mlp.json", "w"))

    # 3) xgboost
    m = json.load(open(RES / "xgb_v2_semilla42.json"))["learner"]
    n = int(m["attributes"]["best_iteration"]) + 1
    p0 = float(m["learner_model_param"]["base_score"].strip("[]"))
    arboles = [{"izq": t["left_children"], "der": t["right_children"], "var": t["split_indices"],
                "valor": r(t["split_conditions"], 8), "falta_izq": t["default_left"]}
               for t in m["gradient_booster"]["model"]["trees"][:n]]
    json.dump({"margen_base": float(np.log(p0 / (1 - p0))), "arboles": arboles},
              open(DESTINO / "xgboost.json", "w"))

    # 4) referencia: percentiles, umbrales y metricas del modelo desplegado
    red, bos = Red(RES / "mlp_v2_semilla42.pt"), Bosque(RES / "xgb_v2_semilla42.json")
    pv_r, pt_r, pv_b, pt_b = red.predecir(Xva), red.predecir(Xte), bos.predecir(Xva), bos.predecir(Xte)
    t_r, t_b = tabla_percentiles(pv_r), tabla_percentiles(pv_b)
    pv_e, pt_e = puntaje_ensamble(pv_r, pv_b, t_r, t_b), puntaje_ensamble(pt_r, pt_b, t_r, t_b)
    yv, yt, montos = va["Class"].to_numpy(), te["Class"].to_numpy(), te["Amount"].to_numpy()
    modelos = {}
    for clave, pv, pt in [("red", pv_r, pt_r), ("xgboost", pv_b, pt_b), ("ensamble", pv_e, pt_e)]:
        u = umbral_por_costo(yv, pv)
        modelos[clave] = {"umbral": u, "validacion": metricas(yv, pv, u, va["Amount"].to_numpy()),
                          "prueba": metricas(yt, pt, u, montos),
                          # para estimar en vivo el efecto de mover el umbral
                          "val_fraudes": r(np.sort(pv[yv == 1]), 9),
                          "val_legitimas": tabla_percentiles(pv[yv == 0])}
    Xtr_raw = tr[["Time"] + COLS_V + ["Amount"]]
    rangos = {c: {"p01": float(Xtr_raw[c].quantile(.01)), "p99": float(Xtr_raw[c].quantile(.99)),
                  "mediana": float(Xtr_raw[c].median())} for c in COLS_V + ["Amount"]}

    # ejemplos para la transaccion individual, tomados del conjunto de prueba
    te_r = te.reset_index(drop=True)
    u_r = modelos["red"]["umbral"]
    def fila(i, nombre):
        x = te_r.loc[i, ["Time"] + COLS_V + ["Amount"]]
        return {"nombre": nombre, "valores": {k: round(float(v), 6) for k, v in x.items()}, "clase": int(te_r.loc[i, "Class"])}
    fr = np.where(yt == 1)[0]; le = np.where(yt == 0)[0]
    ejemplos = [fila(int(le[np.argmin(np.abs(pt_r[le] - np.median(pt_r[le])))]), "Legítima típica"),
                fila(int(fr[np.argmax(pt_r[fr])]), "Fraude detectado con alta confianza"),
                fila(int(fr[np.argsort(pt_r[fr])[len(fr) // 2]]), "Fraude con riesgo intermedio"),
                fila(int(fr[np.argmin(pt_r[fr])]), "Fraude que el modelo no detecta"),
                fila(int(le[np.argmax(pt_r[le])]), "Legítima que genera una falsa alarma")]

    json.dump({"tablas_percentiles": {"red": t_r, "xgboost": t_b, "ensamble": tabla_percentiles(pv_e)}, "modelos": modelos, "rangos": rangos,
               "ejemplos": ejemplos, "costo": {"fn": 1.0, "fp": 0.02},
               "datos": {"validacion": {"filas": int(len(yv)), "fraudes": int(yv.sum())},
                         "prueba": {"filas": int(len(yt)), "fraudes": int(yt.sum())}}},
              open(DESTINO / "referencia.json", "w"), ensure_ascii=False, indent=1)

    # 5) archivos de ejemplo para la carga por lote
    cols = ["Time"] + COLS_V + ["Amount", "Class"]
    fr_pos = np.where(te_r["Class"].to_numpy() == 1)[0]
    ventana = 3000
    inicios = np.arange(0, len(te_r) - ventana, 250)
    mejor = int(max(inicios, key=lambda s: ((fr_pos >= s) & (fr_pos < s + ventana)).sum()))
    lote = te_r.loc[mejor:mejor + ventana - 1, cols].copy()
    lote[COLS_V] = lote[COLS_V].round(6)
    lote.to_csv(EJEMPLOS / "lote_ejemplo.csv", index=False)
    lote.drop(columns="Class").head(3).to_csv(EJEMPLOS / "plantilla.csv", index=False)
    print("lote de ejemplo:", len(lote), "transacciones,", int(lote["Class"].sum()), "fraudes")

    # 6) predicciones de referencia para la prueba de paridad con JavaScript (no se versionan)
    tmp = RAIZ / "tests" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    json.dump({"red": r(pt_r, 12), "xgboost": r(pt_b, 12), "ensamble": r(pt_e, 12)}, open(tmp / "referencia.json", "w"))
    te_r[cols].to_csv(tmp / "prueba.csv", index=False)
    for k, v in modelos.items():
        print(f"{k:9} umbral {v['umbral']:.5f} | prueba", {a: float(b) for a, b in v['prueba'].items()})


if __name__ == "__main__":
    main()
