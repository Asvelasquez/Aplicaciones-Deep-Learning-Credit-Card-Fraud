"""Analisis exploratorio: genera las cuatro figuras del informe.

Esta fase no hace parte del pipeline automatico, es trabajo de
interpretacion, pero sus hallazgos justifican las decisiones del
preprocesamiento.

Uso:
    python notebooks/eda_figuras.py
    python notebooks/eda_figuras.py <ruta_csv> <carpeta_salida>
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE="#2a78d6"; RED="#e34948"; INK="#0b0b0b"; SEC="#52514e"; MUT="#898781"
GRID="#e1e0d9"; SURF="#fcfcfb"
plt.rcParams.update({
 "font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11.5,
 "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
 "axes.edgecolor":"#c3c2b7","axes.labelcolor":SEC,"text.color":INK,
 "xtick.color":MUT,"ytick.color":MUT,"axes.spines.top":False,"axes.spines.right":False,
})
RAIZ = Path(__file__).resolve().parents[1]
csv = sys.argv[1] if len(sys.argv) > 1 else RAIZ / "data/raw/creditcard.csv"
out = sys.argv[2] if len(sys.argv) > 2 else RAIZ / "reports/figures"
Path(out).mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv)
def save(fig,name):
    fig.savefig(f"{out}/{name}.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# --- Fig 1: distribucion de clases (barras horizontales, eje log, etiquetas directas)
fig,ax = plt.subplots(figsize=(7.2,2.3))
vals=[284315,492]; labs=["Legítimas (clase 0)","Fraudulentas (clase 1)"]; cols=[BLUE,RED]
b=ax.barh(labs,vals,color=cols,height=.55)
ax.set_xscale("log"); ax.set_xlim(1,1_200_000)
for r,v,p in zip(b,vals,["99,83 %","0,17 %"]):
    ax.text(v*1.25,r.get_y()+r.get_height()/2,f"{v:,.0f}".replace(",",".")+f"   ({p})",
            va="center",fontsize=10,color=INK)
ax.set_xlabel("Número de transacciones (escala logarítmica)")
ax.xaxis.grid(True,color=GRID,lw=.8); ax.set_axisbelow(True); ax.invert_yaxis()
ax.set_title("Desbalance de clases: 578 transacciones legítimas por cada fraude",loc="left",color=INK,pad=10)
save(fig,"f1_clases")

# --- Fig 2: monto por clase (boxplot log)
fig,ax = plt.subplots(figsize=(6.4,3.2))
d0=df.loc[df.Class==0,"Amount"].clip(lower=.01); d1=df.loc[df.Class==1,"Amount"].clip(lower=.01)
bp=ax.boxplot([d0,d1],orientation="vertical",widths=.45,patch_artist=True,showfliers=False,
              medianprops=dict(color=INK,lw=1.8),whiskerprops=dict(color=MUT),capprops=dict(color=MUT))
for patch,c in zip(bp["boxes"],[BLUE,RED]):
    patch.set_facecolor(c); patch.set_alpha(.35); patch.set_edgecolor(c); patch.set_linewidth(1.6)
ax.set_yscale("log"); ax.set_xticklabels(["Legítimas","Fraudulentas"])
ax.set_ylabel("Monto en euros (escala logarítmica)")
ax.yaxis.grid(True,color=GRID,lw=.8); ax.set_axisbelow(True)
ax.text(1.28,22,"mediana 22,00",fontsize=9,color=SEC,va="center")
ax.text(2.28,9.25,"mediana 9,25",fontsize=9,color=SEC,va="center")
ax.set_title("El monto no separa las clases por sí solo",loc="left",color=INK,pad=10)
save(fig,"f2_monto")

# --- Fig 3: fraudes por hora a lo largo de las 48 horas
fig,ax = plt.subplots(figsize=(7.2,2.8))
h=(df.Time//3600).astype(int); serie=df.groupby(h)["Class"].sum().reindex(range(48),fill_value=0)
ax.plot(serie.index,serie.values,color=RED,lw=2)
ax.fill_between(serie.index,serie.values,color=RED,alpha=.12)
ax.set_xlabel("Horas transcurridas desde la primera transacción")
ax.set_ylabel("Fraudes detectados")
ax.yaxis.grid(True,color=GRID,lw=.8); ax.set_axisbelow(True); ax.set_xlim(0,47)
mx=int(serie.idxmax()); ax.annotate(f"pico de {int(serie.max())} fraudes\nen la hora {mx}",
    xy=(mx,serie.max()),xytext=(mx+3,serie.max()-2),fontsize=9,color=SEC,
    arrowprops=dict(arrowstyle="-",color=MUT,lw=.9))
ax.set_title("Los fraudes se concentran en picos, no se reparten de forma uniforme",loc="left",color=INK,pad=10)
save(fig,"f3_tiempo")

# --- Fig 4: correlacion con la clase (divergente)
fig,ax = plt.subplots(figsize=(6.4,4.2))
corr=df.corr(numeric_only=True)["Class"].drop("Class")
top=corr.reindex(corr.abs().sort_values(ascending=False).index)[:12].sort_values()
cols=[BLUE if v<0 else RED for v in top.values]
ax.barh(top.index,top.values,color=cols,height=.65)
ax.axvline(0,color="#c3c2b7",lw=1)
for i,(n,v) in enumerate(top.items()):
    ax.text(v+(0.012 if v>0 else -0.012),i,f"{v:+.3f}".replace(".",","),va="center",
            ha="left" if v>0 else "right",fontsize=8.5,color=SEC)
ax.set_xlim(-0.44,0.30); ax.set_xlabel("Correlación de Pearson con la variable Class")
ax.xaxis.grid(True,color=GRID,lw=.8); ax.set_axisbelow(True)
ax.set_title("Doce variables más asociadas al fraude\n(azul: relación negativa · rojo: relación positiva)",loc="left",color=INK,pad=10)
save(fig,"f4_corr")
print("ok")
