"""Prueba de paridad: la pagina web debe predecir lo mismo que el modelo en Python.

1. Regenera los archivos del modelo web y las predicciones de referencia en Python.
2. Ejecuta docs/js/modelo.js con Node sobre las 42.559 transacciones de prueba.
3. Exige que ninguna decision de alerta cambie y que la diferencia maxima sea minima.

Uso:
    python tests/test_paridad.py        (o con pytest)
Requiere data/raw/creditcard.csv y Node 18 o superior.
"""
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_paridad():
    subprocess.run([sys.executable, "-m", "src.deploy.exportar_modelo"], cwd=RAIZ, check=True)
    salida = subprocess.run(["node", "tests/paridad.mjs", "tests/tmp/prueba.csv", "tests/tmp/referencia.json"],
                            cwd=RAIZ, check=True, capture_output=True, text=True).stdout
    r = json.loads(salida)
    print(json.dumps(r, indent=2))
    assert r["filas"] == 42559
    assert all(v == 0 for v in r["decisiones_distintas"].values()), "Alguna decision cambio entre Python y JavaScript"
    assert all(v < 1e-4 for v in r["max_diferencia"].values()), "Las probabilidades difieren demasiado"


if __name__ == "__main__":
    test_paridad()
    print("Paridad verificada: JavaScript y Python dan las mismas decisiones.")
