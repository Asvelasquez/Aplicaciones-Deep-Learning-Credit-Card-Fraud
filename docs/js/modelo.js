/*
 * Inferencia del detector de fraude en JavaScript.
 *
 * Reproduce en el navegador el mismo calculo que el modelo entrenado en Python:
 *   1. preprocesamiento: hora circular (seno y coseno), log(1 + monto) y escalado robusto
 *   2. Red MLP v2: 31 -> 128 -> 64 -> 32 -> 1 con BatchNorm y ReLU
 *   3. XGBoost v2: 36 arboles de decision
 *   4. Ensamble: promedio de los percentiles de cada modelo frente a validacion
 *
 * Funciona tanto en el navegador (window.Modelo) como en Node (require), lo que
 * permite verificar con tests/paridad.mjs que da los mismos resultados que Python.
 */
(function (raiz, fabrica) {
  if (typeof module === "object" && module.exports) module.exports = fabrica();
  else raiz.Modelo = fabrica();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const COLUMNAS_ENTRADA = ["Time"].concat(Array.from({ length: 28 }, (_, i) => "V" + (i + 1)), ["Amount"]);

  /* ---------- 1. preprocesamiento ---------- */
  function preprocesar(fila, pre) {
    const x = new Float64Array(31);
    for (let i = 0; i < 28; i++) x[i] = fila["V" + (i + 1)];
    const hora = Math.floor(fila.Time / 3600) % 24;
    x[28] = Math.sin((2 * Math.PI * hora) / 24);
    x[29] = Math.cos((2 * Math.PI * hora) / 24);
    x[30] = Math.log1p(fila.Amount);
    for (let i = 0; i < 31; i++) {
      // Math.fround reproduce el paso a float32 que se hace en Python antes del modelo
      x[i] = Math.fround((x[i] - pre.centro[i]) / pre.escala[i]);
    }
    return x;
  }

  /* ---------- 2. red neuronal ---------- */
  function red(x, R) {
    let h = x;
    for (const c of R.capas) {
      const salida = new Float64Array(c.b.length);
      for (let j = 0; j < c.b.length; j++) {
        let s = c.b[j];
        const w = c.W[j];
        for (let i = 0; i < h.length; i++) s += w[i] * h[i];
        // BatchNorm en modo evaluacion: usa la media y varianza guardadas en el entrenamiento
        s = ((s - c.media[j]) / Math.sqrt(c.var[j] + R.eps)) * c.gamma[j] + c.beta[j];
        salida[j] = s > 0 ? s : 0; // ReLU; el dropout no actua al predecir
      }
      h = salida;
    }
    let z = R.salida.b[0];
    for (let i = 0; i < h.length; i++) z += R.salida.W[i] * h[i];
    return 1 / (1 + Math.exp(-z));
  }

  /* ---------- 3. XGBoost ---------- */
  function xgboost(x, X) {
    let margen = X.margen_base;
    for (const a of X.arboles) {
      let n = 0;
      while (a.izq[n] !== -1) {
        const v = x[a.var[n]];
        const izquierda = Number.isNaN(v) ? a.falta_izq[n] === 1 : v < Math.fround(a.valor[n]);
        n = izquierda ? a.izq[n] : a.der[n];
      }
      margen += a.valor[n]; // en las hojas, "valor" es el aporte del arbol
    }
    return 1 / (1 + Math.exp(-margen));
  }

  /* ---------- 4. percentiles y ensamble ---------- */
  function interpolar(p, tabla) {
    const xs = tabla.valores, ys = tabla.cuantiles;
    if (p <= xs[0]) return ys[0];
    if (p >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (xs[m] <= p) lo = m; else hi = m;
    }
    return ys[lo] + ((ys[hi] - ys[lo]) * (p - xs[lo])) / (xs[hi] - xs[lo]);
  }

  // Inverso de interpolar: dado un cuantil q, devuelve el puntaje que lo alcanza
  function cuantil(q, tabla) {
    return interpolar(q, { valores: tabla.cuantiles, cuantiles: tabla.valores });
  }

  function puntuarVector(x, M) {
    const pr = red(x, M.red);
    const px = xgboost(x, M.xgboost);
    const tp = M.referencia.tablas_percentiles;
    const pe = (interpolar(pr, tp.red) + interpolar(px, tp.xgboost)) / 2;
    return { red: pr, xgboost: px, ensamble: pe };
  }

  function puntuar(fila, M) {
    return puntuarVector(preprocesar(fila, M.pre), M);
  }

  /*
   * Explicacion por oclusion: cada variable se lleva a su valor tipico (0 tras el
   * escalado robusto, que equivale a la mediana) y se mide cuanto cambia el riesgo.
   * La hora se trata como una sola variable aunque entre al modelo como seno y coseno.
   */
  const GRUPOS = Array.from({ length: 28 }, (_, i) => ({ nombre: "V" + (i + 1), idx: [i] }))
    .concat([{ nombre: "Hora", idx: [28, 29] }, { nombre: "Monto", idx: [30] }]);

  function explicar(fila, M, clave) {
    const x = preprocesar(fila, M.pre);
    const base = puntuarVector(x, M)[clave];
    return GRUPOS.map((g) => {
      const x2 = Float64Array.from(x);
      for (const i of g.idx) x2[i] = 0;
      return { variable: g.nombre, aporte: base - puntuarVector(x2, M)[clave] };
    }).sort((a, b) => Math.abs(b.aporte) - Math.abs(a.aporte));
  }

  /* Estimacion, con los datos de validacion, de lo que pasa con un umbral dado */
  function efectoUmbral(clave, umbral, R) {
    const m = R.modelos[clave];
    const detectados = m.val_fraudes.filter((v) => v >= umbral).length;
    const tasaFP = 1 - interpolar(umbral, m.val_legitimas);
    return { recall: detectados / m.val_fraudes.length, falsasPor10k: Math.max(0, tasaFP) * 10000 };
  }

  async function cargar(base) {
    const leer = async (n) => (await fetch(base + n)).json();
    const [pre, r, x, ref] = await Promise.all([
      leer("preprocesamiento.json"), leer("red_mlp.json"), leer("xgboost.json"), leer("referencia.json")]);
    return { pre, red: r, xgboost: x, referencia: ref };
  }

  return { COLUMNAS_ENTRADA, preprocesar, red, xgboost, interpolar, cuantil, puntuar, puntuarVector,
           explicar, efectoUmbral, cargar };
});
