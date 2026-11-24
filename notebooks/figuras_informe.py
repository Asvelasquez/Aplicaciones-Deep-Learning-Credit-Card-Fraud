"""Figuras del informe de los Avances III y IV a partir de los resultados de Colab.

Uso:
    python notebooks/figuras_informe.py resultados/avances_3_4 resultados/avances_3_4/figuras
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve

R = sys.argv[1]; OUT = sys.argv[2]
res = json.load(open(f"{R}/resultados.json", encoding="utf-8"))
pred = np.load(f"{R}/predicciones.npz"); etq = np.load(f"{R}/etiquetas.npz")
y_va, y_te, montos = etq["y_val"], etq["y_test"], etq["montos_test"]
M = res["modelos"]; H = res["historias"]; X = res["extra"]

INK, SEC, MUT, GRID = "#1F2328", "#4B5563", "#8A8F98", "#E5E7EB"
C = {"logreg": "#8A8F98", "xgb_v1": "#119C8B", "xgb_v2": "#119C8B", "lgb_v1": "#7FCFC4",
     "mlp_v1": "#E58A00", "mlp_v2": "#E58A00", "tabnet_v1": "#7E5BEF", "mlp_emb": "#D6457D", "ensamble": "#4E9A06"}
NOM = {"logreg": "Regresión logística", "xgb_v1": "XGBoost v1", "lgb_v1": "LightGBM v1", "mlp_v1": "Red MLP v1",
       "tabnet_v1": "TabNet v1", "xgb_v2": "XGBoost v2", "mlp_v2": "Red MLP v2", "mlp_emb": "MLP v2 + embeddings",
       "ensamble": "Ensamble XGB + MLP"}
plt.rcParams.update({"font.family": "Inter", "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": "#D1D5DB", "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True})
coma = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
def guardar(fig, n): fig.savefig(f"{OUT}/{n}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- G1: curvas PR v1 en paneles pequeños + resumen val/prueba
v1 = ["logreg", "xgb_v1", "lgb_v1", "mlp_v1", "tabnet_v1"]
curvas = {k: precision_recall_curve(y_te, pred[f"{k}__test"]) for k in v1}
fig, ejes = plt.subplots(2, 3, figsize=(10, 6.2))
for ax, k in zip(ejes.flat, v1):
    for o in v1:
        if o != k: ax.plot(curvas[o][1], curvas[o][0], color="#D1D5DB", lw=1)
    ax.plot(curvas[k][1], curvas[k][0], color=C[k], lw=2.4)
    ax.set_title(NOM[k], loc="left"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.text(.04, .08, f"AUPRC prueba {coma(M[k]['test']['auprc'])}", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#D1D5DB"))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precisión")
ax = ejes.flat[5]
orden = sorted(v1, key=lambda k: M[k]["test"]["auprc"])
for i, k in enumerate(orden):
    a, b = M[k]["val"]["auprc"], M[k]["test"]["auprc"]
    ax.plot([a, b], [i, i], color="#D1D5DB", lw=2, zorder=1)
    ax.scatter(a, i, s=46, facecolor="white", edgecolor=C[k], lw=1.8, zorder=2)
    ax.scatter(b, i, s=46, color=C[k], zorder=3)
ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOM[k] for k in orden], fontsize=8.5)
ax.set_xlim(0, 1); ax.set_title("Validación (○) frente a prueba (●)", loc="left"); ax.set_xlabel("AUPRC"); ax.grid(axis="y", visible=False)
fig.tight_layout(h_pad=2); guardar(fig, "g1_pr_v1")

# ---------- G2: dinamica de entrenamiento en tres paneles apilados
h = H["mlp_v1"]; ep = np.arange(1, len(h["val_loss"]) + 1)
e_loss = h["mejor_epoca"]; e_auc = int(np.argmax(h["val_auprc"])) + 1
fig, ejes = plt.subplots(3, 1, figsize=(8.6, 8.2), gridspec_kw={"height_ratios": [1, 1, 1.1]})
ax = ejes[0]; ax.plot(ep, h["val_loss"], color=C["mlp_v1"], lw=2); ax.set_yscale("log")
ax.axvline(e_loss, color=SEC, ls="--", lw=1); ax.text(e_loss + .4, max(h["val_loss"]) * .5, f"mínimo de pérdida\népoca {e_loss}", fontsize=8.5, color=SEC)
ax.set_title("a) Red MLP v1, pérdida focal en validación", loc="left"); ax.set_ylabel("Pérdida (log)"); ax.set_xlim(1, len(ep))
ax = ejes[1]; ax.plot(ep, h["val_auprc"], color=C["mlp_v1"], lw=2)
ax.axvline(e_loss, color=SEC, ls="--", lw=1); ax.axvline(e_auc, color=C["mlp_v1"], ls=":", lw=1.4)
ax.scatter([e_auc], [max(h["val_auprc"])], color=C["mlp_v1"], zorder=3, s=36)
ax.text(e_auc + .4, .74, f"máximo AUPRC\népoca {e_auc}: {coma(max(h['val_auprc']))}", fontsize=8.5, color=INK)
ax.text(e_loss + .4, .74, f"época elegida por pérdida\nAUPRC {coma(h['val_auprc'][e_loss-1])}", fontsize=8.5, color=SEC)
ax.set_ylim(.7, .9); ax.set_xlim(1, len(ep)); ax.set_title("b) Red MLP v1, AUPRC en validación", loc="left"); ax.set_ylabel("AUPRC"); ax.set_xlabel("Época")
ax = ejes[2]
hx = H["xgb_v1"]["val_aucpr"]; hl = H["lgb_v1"]["val_aucpr"]
ax.plot(np.arange(1, len(hx) + 1), hx, color=C["xgb_v1"], lw=2, label="XGBoost v1")
ax.plot(np.arange(1, len(hl) + 1), hl, color="#4FB6A8", lw=2, ls="--", label="LightGBM v1")
ax.annotate("LightGBM alcanza 0,75 en la iteración 5\ny luego se degrada", xy=(5, max(hl)), xytext=(40, .25),
            fontsize=8.5, color=SEC, arrowprops=dict(arrowstyle="-", color=MUT))
ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_title("c) Boosting, AUPRC en validación por iteración", loc="left")
ax.set_xlabel("Iteración (escala log)"); ax.set_ylabel("AUPRC"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(h_pad=1.6); guardar(fig, "g2_dinamica")

# ---------- G3: importancia de variables, SHAP de XGBoost frente a TabNet (dumbbell)
sh = X["shap_media_abs"]; tb = X["importancia_tabnet"]
shp = {k: 100 * v / sum(sh.values()) for k, v in sh.items()}; tbp = {k: 100 * v / sum(tb.values()) for k, v in tb.items()}
top = sorted(set(sorted(shp, key=shp.get, reverse=True)[:8]) | set(sorted(tbp, key=tbp.get, reverse=True)[:8]),
             key=lambda k: shp[k] + tbp[k])
fig, ax = plt.subplots(figsize=(7.4, 5.4))
for i, k in enumerate(top):
    ax.plot([shp[k], tbp[k]], [i, i], color="#D1D5DB", lw=2.2, zorder=1)
    ax.scatter(shp[k], i, color=C["xgb_v1"], s=55, zorder=2, label="XGBoost (SHAP)" if i == 0 else None)
    ax.scatter(tbp[k], i, color=C["tabnet_v1"], marker="D", s=46, zorder=3, label="TabNet (máscaras de atención)" if i == 0 else None)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top); ax.grid(axis="y", visible=False)
ax.set_xlabel("Participación en la importancia total (%)")
ax.legend(frameon=False, loc="lower right"); ax.set_title("Variables más importantes según cada modelo", loc="left")
fig.tight_layout(); guardar(fig, "g3_importancia")

# ---------- G4: fraudes de prueba, detectados y no detectados (MLP v1)
fn = X["analisis_fn"]; u = fn["umbral"]; fr = fn["fraudes"]
prob = np.array([f["prob"] for f in fr]); mto = np.array([f["monto"] for f in fr]); det = np.array([f["detectado"] for f in fr])
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [3.2, 1]})
rng = np.random.default_rng(3)
for fila, (m, nombre, col) in enumerate([(det, "Detectados", C["mlp_v1"]), (~det, "No detectados", INK)]):
    ax.scatter(np.clip(prob[m], 1e-6, 1), fila + rng.uniform(-.18, .18, m.sum()), s=18 + np.sqrt(mto[m]) * 6,
               facecolor=col if fila == 0 else "white", edgecolor=col, lw=1.3, alpha=.85)
ax.axvline(u, color=SEC, ls="--", lw=1); ax.text(u * 1.15, 1.42, f"umbral {coma(u, 3)}", fontsize=8.5, color=SEC)
ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([0.04, 0.06, 0.1, 0.2, 0.3, 0.5, 0.8]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
ax.set_yticks([0, 1]); ax.set_yticklabels([f"Detectados ({det.sum()})", f"No detectados ({(~det).sum()})"])
ax.set_ylim(-.6, 1.65); ax.grid(axis="y", visible=False)
ax.set_xlabel("Probabilidad de fraude asignada por la red (escala log), tamaño del punto = monto")
ax.set_title("Los 52 fraudes del conjunto de prueba", loc="left")
tot_d, tot_n = mto[det].sum(), mto[~det].sum()
ax2.barh([0], [tot_d], color=C["mlp_v1"], height=.5); ax2.barh([0], [tot_n], left=[tot_d], color="white", edgecolor=INK, height=.5)
ax2.text(tot_d / 2, 0, f"€{tot_d:,.0f}".replace(",", "."), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax2.text(tot_d + tot_n / 2, 0, f"€{tot_n:,.0f}".replace(",", "."), ha="center", va="center", color=INK, fontsize=9, fontweight="bold")
ax2.set_yticks([]); ax2.set_xlim(0, (tot_d + tot_n) * 1.02); ax2.set_title("Monto en juego", loc="left"); ax2.set_xlabel("Euros"); ax2.grid(visible=False)
ax2.text(0, -.55, f"{100*tot_n/(tot_d+tot_n):.0f} % del monto está en los no detectados".replace(".", ","), fontsize=8.3, color=SEC)
fig.tight_layout(); guardar(fig, "g4_errores")

# ---------- G7: forest plot de AUPRC con IC para todos los modelos
orden = sorted(M, key=lambda k: M[k]["test"]["auprc"])
fig, ax = plt.subplots(figsize=(8, 4.8))
for i, k in enumerate(orden):
    m = M[k]["test"]["auprc"]; lo, hi = M[k]["ic_test"]["auprc_ic95"]
    ax.plot([lo, hi], [i, i], color=C[k], lw=2.2, solid_capstyle="round")
    v2 = M[k].get("fase") == "v2"
    ax.scatter(m, i, s=60, color=C[k] if v2 else "white", edgecolor=C[k], lw=2, zorder=3)
    ax.text(1.005, i, coma(m), va="center", fontsize=8.8, color=INK)
ax.axvline(M["logreg"]["test"]["auprc"], color=MUT, ls="--", lw=1)
ax.text(M["logreg"]["test"]["auprc"] + .005, -.9, "línea base", fontsize=8, color=MUT)
ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOM[k] for k in orden]); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 1); ax.set_ylim(-1.2, len(orden) - .4); ax.set_xlabel("AUPRC en prueba con intervalo de confianza del 95 %")
ax.set_title("Todos los modelos, v1 (○) y v2 (●)", loc="left")
fig.tight_layout(); guardar(fig, "g7_forest")

# ---------- G8: diferencias pareadas de AUPRC (bootstrap)
etiquetas = {"xgb_v1 vs mlp_v1": "XGBoost v1 − MLP v1", "xgb_v2 vs mlp_v2": "XGBoost v2 − MLP v2",
             "xgb_v2 vs mlp_emb": "XGBoost v2 − MLP + embeddings", "ensamble vs xgb_v2": "Ensamble − XGBoost v2",
             "xgb_v2 vs xgb_v1": "XGBoost v2 − XGBoost v1", "mlp_v2 vs mlp_v1": "MLP v2 − MLP v1",
             "mlp_emb vs mlp_v2": "MLP + embeddings − MLP v2", "xgb_v1 vs logreg": "XGBoost v1 − Regresión logística"}
comp = X["comparaciones"]; claves = list(etiquetas)[::-1]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
for i, k in enumerate(claves):
    d = comp[k]; lo, hi = d["ic95"]; m = d["diferencia_media"]
    sig = lo > 0 or hi < 0
    ax.plot([lo, hi], [i, i], color=INK if sig else "#9CA3AF", lw=3, solid_capstyle="round", alpha=.9)
    ax.scatter(m, i, color=INK, s=42, zorder=3)
    ax.text(.158, i, f"P(mejor) {coma(100*d['prob_a_mejor'],0)} %", va="center", fontsize=8.3, color=SEC)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(claves))); ax.set_yticklabels([etiquetas[k] for k in claves]); ax.grid(axis="y", visible=False)
ax.set_xlim(-.16, .15); ax.set_xlabel("Diferencia de AUPRC en prueba (A − B) con IC 95 %")
ax.set_title("Comparaciones pareadas: ningún intervalo excluye el cero", loc="left")
fig.tight_layout(); guardar(fig, "g8_pareadas")

# ---------- G9: punto de operación en tres paneles
fin = ["mlp_v1", "xgb_v2", "mlp_v2", "mlp_emb", "ensamble"]
fig, ejes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
yy = np.arange(len(fin))[::-1]
for j, k in enumerate(fin):
    t = M[k]["test"]
    ejes[0].barh(yy[j], t["vp"], color=C[k], height=.6); ejes[0].barh(yy[j], t["fn"], left=t["vp"], color="white", edgecolor=C[k], height=.6)
    ejes[0].text(t["vp"] - 1, yy[j], f"{int(t['vp'])}", ha="right", va="center", color="white", fontsize=8.5, fontweight="bold")
    ejes[1].barh(yy[j], t["fp"], color=C[k], height=.6); ejes[1].text(t["fp"] + 1.5, yy[j], f"{int(t['fp'])}", va="center", fontsize=8.5)
    ejes[2].barh(yy[j], t["f1"], color=C[k], height=.6); ejes[2].text(t["f1"] + .01, yy[j], coma(t["f1"]), va="center", fontsize=8.5)
ejes[0].set_yticks(yy); ejes[0].set_yticklabels([NOM[k] for k in fin])
ejes[0].set_title("Fraudes detectados de 52", loc="left"); ejes[0].set_xlim(0, 52)
ejes[1].set_title("Alertas falsas", loc="left"); ejes[1].set_xlim(0, 100)
ejes[2].set_title("F1 en el umbral elegido", loc="left"); ejes[2].set_xlim(0, .8)
for ax in ejes: ax.grid(axis="y", visible=False)
fig.tight_layout(); guardar(fig, "g9_operacion")

# ---------- G10: latencia frente a AUPRC
fig, ax = plt.subplots(figsize=(7.4, 4.3))
desp = {"logreg": (6, -10), "xgb_v1": (6, 5), "lgb_v1": (6, 4), "mlp_v1": (-8, -13), "mlp_v2": (6, 6),
        "xgb_v2": (-8, -13), "mlp_emb": (0, -17)}
for k in desp:
    lat, a = M[k]["latencia_ms"], M[k]["test"]["auprc"]
    ax.scatter(lat, a, s=70, color=C[k] if M[k].get("fase") == "v2" else "white", edgecolor=C[k], lw=2, zorder=3)
    ha = "right" if desp[k][0] < 0 else ("center" if desp[k][0] == 0 else "left")
    ax.annotate(NOM[k], (lat, a), textcoords="offset points", xytext=desp[k], fontsize=8.5, ha=ha, color=INK)
ax.set_xscale("log"); ax.set_xlim(.25, 30); ax.set_ylim(0, .9)
ax.set_xlabel("Milisegundos para puntuar 1.000 transacciones en CPU (escala log)"); ax.set_ylabel("AUPRC en prueba")
ax.set_title("Costo computacional frente a desempeño", loc="left")
fig.tight_layout(); guardar(fig, "g10_latencia")
print("ok")

# ---------- G6: porcentaje de avance por tarea
tareas = [("Definición del problema y revisión de literatura", 10, 1.0), ("Diseño de la solución", 10, 1.0),
          ("Pipeline de datos y repositorio", 15, 1.0), ("Implementación de los modelos v1", 15, 1.0),
          ("Pruebas iniciales y análisis de resultados", 15, 1.0), ("Rediseño implementado (5 de 6 cambios)", 15, 5 / 6),
          ("Comparación estadística final", 10, 1.0), ("Entregas finales (Avances V y VI)", 10, 0.0)]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for i, (t, w, f) in enumerate(tareas[::-1]):
    ax.barh(i, w, color="white", edgecolor="#9CA3AF", height=.62, lw=1)
    if f > 0: ax.barh(i, w * f, color="#119C8B" if f == 1 else "#E58A00", height=.62)
    ax.text(w + .3, i, f"{coma(w*f,1)} de {w}".replace(",0 ", " "), va="center", fontsize=8.5, color=SEC)
ax.set_yticks(range(len(tareas))); ax.set_yticklabels([t for t, _, _ in tareas[::-1]]); ax.grid(axis="y", visible=False)
total = sum(w * f for _, w, f in tareas)
ax.set_xlim(0, 18); ax.set_xlabel("Puntos de avance (el peso de cada tarea suma 100)")
ax.set_title(f"Avance total del proyecto: {coma(total,1)} %", loc="left")
fig.tight_layout(); guardar(fig, "g6_avance")
print("total avance", total)
