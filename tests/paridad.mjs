// Verifica que la inferencia en JavaScript da lo mismo que la referencia en Python.
// Uso: node tests/paridad.mjs <csv de prueba> <predicciones de referencia en JSON>
import fs from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const Modelo = require("../docs/js/modelo.js");
const leer = (n) => JSON.parse(fs.readFileSync(new URL("../docs/modelo/" + n, import.meta.url)));
const M = { pre: leer("preprocesamiento.json"), red: leer("red_mlp.json"), xgboost: leer("xgboost.json"), referencia: leer("referencia.json") };

const [csv, refJson] = process.argv.slice(2);
const lineas = fs.readFileSync(csv, "utf8").trim().split(/\r?\n/);
const cab = lineas[0].split(",").map((c) => c.replace(/"/g, ""));
const ref = JSON.parse(fs.readFileSync(refJson));
const dif = { red: 0, xgboost: 0, ensamble: 0 };
const decisionesDistintas = { red: 0, xgboost: 0, ensamble: 0 };
const t0 = Date.now();
for (let k = 1; k < lineas.length; k++) {
  const v = lineas[k].split(",");
  const fila = {};
  cab.forEach((c, i) => (fila[c] = parseFloat(v[i].replace(/"/g, ""))));
  const p = Modelo.puntuar(fila, M);
  for (const c of Object.keys(dif)) {
    dif[c] = Math.max(dif[c], Math.abs(p[c] - ref[c][k - 1]));
    const u = M.referencia.modelos[c].umbral;
    if ((p[c] >= u) !== (ref[c][k - 1] >= u)) decisionesDistintas[c]++;
  }
}
console.log(JSON.stringify({ filas: lineas.length - 1, max_diferencia: dif, decisiones_distintas: decisionesDistintas,
                             segundos: (Date.now() - t0) / 1000 }));
