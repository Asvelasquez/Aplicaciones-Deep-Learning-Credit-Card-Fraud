/*
 * Interfaz del detector de fraude.
 * Toda la logica de prediccion esta en modelo.js; este archivo solo maneja la
 * pagina: carga de archivos, tabla de alertas, formulario individual y graficas.
 */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const num = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 0 });
  const dec = (d) => new Intl.NumberFormat("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
  const euros = new Intl.NumberFormat("es-ES", { style: "currency", currency: "EUR", maximumFractionDigits: 2 });
  const MAX_FILAS = 300000;
  const FILAS_TABLA = 500;
  const DESLIZADORES = ["V14", "V4", "V12", "V10", "V11", "V17"];
  const NOMBRES = { red: "Red neuronal", xgboost: "XGBoost", ensamble: "Ensamble" };
  const COLORES = { red: "var(--red)", xgboost: "var(--xgboost)", ensamble: "var(--ensamble)" };
  const DESCRIPCION = {
    red: "Red MLP v2, el modelo de deep learning del proyecto. Es el que más fraudes detecta.",
    xgboost: "Gradient Boosting con 36 árboles de decisión. Rinde parecido a la red.",
    ensamble: "Combina la red y XGBoost. Detecta casi los mismos fraudes con la mitad de alertas falsas."
  };

  let M = null;                 // modelos cargados
  let clave = "red";            // modelo elegido
  const umbral = {};            // umbral actual por modelo
  let lote = null;              // resultados del ultimo archivo
  let actual = null;            // transaccion del formulario individual

  /* ---------- escalas del umbral y del riesgo ---------- */
  // Las salidas de los modelos no son probabilidades calibradas: la red se entreno con
  // perdida focal y XGBoost con un peso grande para los fraudes. Por eso el riesgo se
  // muestra como un puntaje de 0 a 100 y el umbral se mueve en terminos de "que porcentaje
  // de las transacciones mas riesgosas se envia a revision", medido en validacion.
  const tabla = (c) => M.referencia.tablas_percentiles[c];
  const percentil = (c, v) => Modelo.interpolar(v, tabla(c));
  const escala = {
    aUmbral(c, s) {
      const q = 1 - Math.pow(10, -(1 + 4 * s / 1000));   // de marcar el 10 % a marcar el 0,001 %
      return Modelo.cuantil(q, tabla(c));
    },
    aDeslizador(c, u) {
      const k = -Math.log10(Math.max(1 - percentil(c, u), 1e-6));
      return Math.round(Math.min(1000, Math.max(0, ((k - 1) / 4) * 1000)));
    },
    // posicion visual entre 0 y 1 para barras y medidores, igual para los tres modelos
    visual(c, v) {
      return Math.min(1, Math.max(0, -Math.log10(Math.max(1 - percentil(c, v), 1e-6)) / 5));
    }
  };
  const fmtRiesgo = (c, v) => dec(c === "ensamble" ? 2 : 1).format(100 * v);
  const fmtUmbral = (c, u) => `marca el ${dec(2).format(100 * (1 - percentil(c, u)))} % más riesgoso · puntaje ≥ ${fmtRiesgo(c, u)}`;
  const horaDe = (t) => Math.floor(t / 3600) % 24;

  /* ---------- inicio ---------- */
  async function iniciar() {
    configurarPestanas();
    try {
      M = await Modelo.cargar("modelo/");
    } catch (e) {
      aviso("No se pudieron cargar los modelos. Si abriste el archivo directamente, sírvelo con un servidor local (ver README).", true);
      return;
    }
    for (const c of Object.keys(NOMBRES)) umbral[c] = M.referencia.modelos[c].umbral;
    document.querySelectorAll('input[name="modelo"]').forEach((r) => r.addEventListener("change", (e) => cambiarModelo(e.target.value)));
    $("#umbral").addEventListener("input", (e) => { umbral[clave] = escala.aUmbral(clave, +e.target.value); refrescar(); });
    $("#umbral-reset").addEventListener("click", () => { umbral[clave] = M.referencia.modelos[clave].umbral; refrescar(true); });
    configurarCarga();
    configurarFormulario();
    llenarInfo();
    enlaceRepositorio();
    cambiarModelo("red");
  }

  function cambiarModelo(c) {
    clave = c;
    $("#desc-modelo").textContent = DESCRIPCION[c];
    refrescar(true);
  }

  function refrescar(moverDeslizador) {
    const u = umbral[clave];
    if (moverDeslizador) $("#umbral").value = escala.aDeslizador(clave, u);
    $("#umbral-valor").textContent = fmtUmbral(clave, u);
    const ef = Modelo.efectoUmbral(clave, u, M.referencia);
    const recomendado = Math.abs(u - M.referencia.modelos[clave].umbral) < 1e-9;
    $("#efecto-umbral").textContent = `${recomendado ? "Umbral recomendado. " : ""}En validación, este umbral detectaba el ${dec(0).format(100 * ef.recall)} % de los fraudes con unas ${dec(1).format(ef.falsasPor10k)} alertas falsas por cada 10.000 transacciones.`;
    if (lote) mostrarLote();
    if (actual) evaluarActual();
  }

  /* ---------- pestañas ---------- */
  function configurarPestanas() {
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    const activar = (t) => {
      tabs.forEach((x) => {
        const on = x === t;
        x.setAttribute("aria-selected", on);
        x.tabIndex = on ? 0 : -1;
        document.getElementById(x.getAttribute("aria-controls")).hidden = !on;
      });
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => activar(t));
      t.addEventListener("keydown", (e) => {
        const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (d) { const n = tabs[(i + d + tabs.length) % tabs.length]; activar(n); n.focus(); }
      });
    });
    window.irA = (id) => activar(document.getElementById(id));
  }

  /* ---------- carga del archivo ---------- */
  function configurarCarga() {
    const zona = $("#zona-carga");
    $("#archivo").addEventListener("change", (e) => e.target.files[0] && leerArchivo(e.target.files[0]));
    $("#usar-ejemplo").addEventListener("click", async () => {
      const texto = await (await fetch("ejemplos/lote_ejemplo.csv")).text();
      procesarTexto(texto, "lote_ejemplo.csv");
    });
    zona.addEventListener("keydown", (e) => { if (e.key === "Enter") $("#archivo").click(); });
    ["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("sobre"); }));
    ["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("sobre"); }));
    zona.addEventListener("drop", (e) => e.dataTransfer.files[0] && leerArchivo(e.dataTransfer.files[0]));
    $("#ver-todas").addEventListener("change", mostrarLote);
    $("#descargar").addEventListener("click", descargar);
  }

  async function leerArchivo(archivo) {
    if (!/\.csv$/i.test(archivo.name)) return aviso("El archivo debe ser un CSV (separado por comas).", true);
    if (archivo.size > 200 * 1024 * 1024) return aviso("El archivo supera 200 MB. Divídelo en partes más pequeñas.", true);
    procesarTexto(await archivo.text(), archivo.name);
  }

  function aviso(texto, error) {
    const a = $("#avisos");
    a.hidden = !texto;
    a.className = "avisos" + (error ? " error" : "");
    a.textContent = texto || "";
  }

  function procesarTexto(texto, nombre) {
    aviso("");
    const lineas = texto.split(/\r?\n/).filter((l) => l.trim() !== "");
    if (lineas.length < 2) return aviso("El archivo está vacío o solo tiene encabezado.", true);
    const cab = lineas[0].split(",").map((c) => c.replace(/"/g, "").trim());
    const pos = {};
    cab.forEach((c, i) => (pos[c.toLowerCase()] = i));
    const faltan = Modelo.COLUMNAS_ENTRADA.filter((c) => !(c.toLowerCase() in pos));
    if (faltan.length) return aviso(`Faltan columnas obligatorias: ${faltan.join(", ")}. Revisa la plantilla.`, true);
    const n = lineas.length - 1;
    if (n > MAX_FILAS) return aviso(`El archivo tiene ${num.format(n)} filas; el máximo es ${num.format(MAX_FILAS)}.`, true);

    const idx = Modelo.COLUMNAS_ENTRADA.map((c) => pos[c.toLowerCase()]);
    const iClase = pos["class"];
    const datos = { nombre, filas: [], valores: new Float64Array(n * 30), errores: [], tieneClase: iClase !== undefined };
    $("#estado").hidden = false;
    $("#resultados-lote").hidden = true;
    let k = 1;
    const t0 = performance.now();

    function bloque() {
      const fin = Math.min(k + 3000, lineas.length);
      for (; k < fin; k++) {
        const v = lineas[k].split(",");
        const fila = {};
        let ok = v.length >= cab.length;
        for (let j = 0; j < 30 && ok; j++) {
          const x = parseFloat((v[idx[j]] || "").replace(/"/g, ""));
          if (!Number.isFinite(x)) ok = false;
          fila[Modelo.COLUMNAS_ENTRADA[j]] = x;
        }
        if (ok && (fila.Amount < 0 || fila.Time < 0)) ok = false;
        if (!ok) { datos.errores.push(k + 1); continue; }
        const i = datos.filas.length;
        for (let j = 0; j < 30; j++) datos.valores[i * 30 + j] = fila[Modelo.COLUMNAS_ENTRADA[j]];
        const p = Modelo.puntuar(fila, M);
        const clase = datos.tieneClase ? parseFloat((v[iClase] || "").replace(/"/g, "")) : null;
        datos.filas.push({ linea: k + 1, tiempo: fila.Time, monto: fila.Amount, clase, p });
      }
      $("#progreso").style.width = (100 * k / lineas.length).toFixed(1) + "%";
      $("#estado-texto").textContent = `Analizando ${nombre}: ${num.format(k - 1)} de ${num.format(n)} transacciones…`;
      if (k < lineas.length) return requestAnimationFrame(bloque);
      const seg = (performance.now() - t0) / 1000;
      $("#estado-texto").textContent = `${nombre}: ${num.format(datos.filas.length)} transacciones analizadas en ${dec(1).format(seg)} s, sin salir de este equipo.`;
      if (datos.errores.length) {
        aviso(`Se omitieron ${num.format(datos.errores.length)} filas con valores vacíos, no numéricos o negativos (por ejemplo, las líneas ${datos.errores.slice(0, 5).join(", ")}).`);
      }
      if (!datos.filas.length) return aviso("Ninguna fila tenía datos válidos.", true);
      lote = datos;
      mostrarLote();
    }
    requestAnimationFrame(bloque);
  }

  /* ---------- resultados del lote ---------- */
  function mostrarLote() {
    const u = umbral[clave];
    const filas = lote.filas;
    const alertas = filas.filter((f) => f.p[clave] >= u);
    const montoAlerta = alertas.reduce((s, f) => s + f.monto, 0);
    $("#resultados-lote").hidden = false;
    $("#kpis").innerHTML = [
      kpi("Transacciones analizadas", num.format(filas.length), ""),
      kpi("Alertas", num.format(alertas.length), "con " + NOMBRES[clave].toLowerCase(), true),
      kpi("Proporción en alerta", dec(2).format(100 * alertas.length / filas.length) + " %", "del lote"),
      kpi("Monto en alerta", euros.format(montoAlerta), "a revisar")
    ].join("");

    const reales = $("#kpis-reales");
    reales.hidden = !lote.tieneClase;
    $("#th-real").hidden = !lote.tieneClase;
    if (lote.tieneClase) {
      const fraudes = filas.filter((f) => f.clase === 1);
      const vp = alertas.filter((f) => f.clase === 1).length;
      const fp = alertas.length - vp;
      reales.innerHTML = '<div class="kpis-titulo">El archivo trae la columna Class, así que se puede medir el acierto real</div>' + [
        kpi("Fraudes reales en el lote", num.format(fraudes.length), dec(3).format(100 * fraudes.length / filas.length) + " % del total"),
        kpi("Fraudes detectados", `${vp} de ${fraudes.length}`, fraudes.length ? dec(0).format(100 * vp / fraudes.length) + " % (recall)" : ""),
        kpi("Precisión de las alertas", alertas.length ? dec(0).format(100 * vp / alertas.length) + " %" : "–", "alertas que son fraude"),
        kpi("Alertas falsas", num.format(fp), "transacciones legítimas marcadas")
      ].join("");
    }
    graficaHoras(alertas, filas);

    const todas = $("#ver-todas").checked;
    const lista = (todas ? filas.slice() : alertas).sort((a, b) => b.p[clave] - a.p[clave]);
    $("#titulo-tabla").textContent = todas ? "Todas las transacciones" : "Transacciones en alerta";
    const cuerpo = $("#tabla tbody");
    cuerpo.innerHTML = lista.slice(0, FILAS_TABLA).map((f, i) => {
      const v = f.p[clave];
      const al = v >= u;
      return `<tr tabindex="0" data-i="${filas.indexOf(f)}">
        <td>${i + 1}</td><td>${f.linea}</td><td>${horaDe(f.tiempo)} h</td><td class="num">${dec(2).format(f.monto)}</td>
        <td><div class="riesgo"><div class="riesgo-barra"><div style="width:${(100 * escala.visual(clave, v)).toFixed(1)}%"></div></div>${fmtRiesgo(clave, v)}</div></td>
        <td><span class="chip ${al ? "alerta" : "normal"}">${al ? "Alerta" : "Normal"}</span></td>
        ${lote.tieneClase ? `<td><span class="chip ${f.clase === 1 ? "fraude" : "legitima"}">${f.clase === 1 ? "Fraude" : "Legítima"}</span></td>` : ""}
      </tr>`;
    }).join("") || `<tr><td colspan="7">Ninguna transacción supera el umbral actual.</td></tr>`;
    $("#pie-tabla").textContent = lista.length > FILAS_TABLA ? `Se muestran las ${FILAS_TABLA} de mayor riesgo de ${num.format(lista.length)}. El CSV descargable las incluye todas.` : "";
    cuerpo.querySelectorAll("tr[data-i]").forEach((tr) => {
      const abrir = () => abrirFila(+tr.dataset.i);
      tr.addEventListener("click", abrir);
      tr.addEventListener("keydown", (e) => e.key === "Enter" && abrir());
    });
  }

  const kpi = (et, val, nota, dest) =>
    `<div class="kpi${dest ? " destacado" : ""}"><div class="kpi-etiqueta">${et}</div><div class="kpi-valor">${val}</div><div class="kpi-nota">${nota}</div></div>`;

  function graficaHoras(alertas, filas) {
    const porHora = new Array(24).fill(0), fraudes = new Array(24).fill(0);
    alertas.forEach((f) => porHora[horaDe(f.tiempo)]++);
    if (lote.tieneClase) filas.forEach((f) => f.clase === 1 && fraudes[horaDe(f.tiempo)]++);
    const max = Math.max(1, ...porHora, ...fraudes);
    const W = 720, H = 190, iz = 34, ab = 24, ar = 10, ancho = (W - iz - 8) / 24;
    const y = (v) => ar + (H - ar - ab) * (1 - v / max);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Alertas por hora del día">`;
    for (const t of [0, max / 2, max]) {
      s += `<line x1="${iz}" x2="${W - 8}" y1="${y(t)}" y2="${y(t)}" stroke="#EEF0F2"/><text x="${iz - 6}" y="${y(t) + 4}" text-anchor="end">${Math.round(t)}</text>`;
    }
    porHora.forEach((v, h) => {
      const x = iz + h * ancho + 3;
      s += `<rect x="${x}" y="${y(v)}" width="${ancho - 6}" height="${H - ab - y(v)}" rx="3" fill="#D9534F"><title>${h} h: ${v} alertas${lote.tieneClase ? ", " + fraudes[h] + " fraudes reales" : ""}</title></rect>`;
      if (lote.tieneClase && fraudes[h]) {
        const cx = x + (ancho - 6) / 2, cy = y(fraudes[h]);
        s += `<path d="M${cx} ${cy - 6} L${cx + 6} ${cy} L${cx} ${cy + 6} L${cx - 6} ${cy} Z" fill="#6941C6" stroke="#fff" stroke-width="1.5"/>`;
      }
      if (h % 2 === 0) s += `<text x="${x + (ancho - 6) / 2}" y="${H - 6}" text-anchor="middle">${h}</text>`;
    });
    $("#grafica-horas").innerHTML = s + "</svg>";
    $("#leyenda-reales").textContent = lote.tieneClase ? "; los rombos morados indican cuántos fraudes reales hubo en esa hora" : "";
  }

  function descargar() {
    if (!lote) return;
    const u = umbral[clave];
    const cab = Modelo.COLUMNAS_ENTRADA.concat(lote.tieneClase ? ["Class"] : [],
      ["puntaje_red", "puntaje_xgboost", "puntaje_ensamble", "modelo_usado", "umbral", "alerta"]);
    const lineas = [cab.join(",")];
    lote.filas.forEach((f, i) => {
      const vals = Array.from(lote.valores.subarray(i * 30, i * 30 + 30));
      if (lote.tieneClase) vals.push(f.clase);
      vals.push(f.p.red.toFixed(6), f.p.xgboost.toFixed(6), f.p.ensamble.toFixed(6), clave, u.toFixed(6), f.p[clave] >= u ? 1 : 0);
      lineas.push(vals.join(","));
    });
    const url = URL.createObjectURL(new Blob([lineas.join("\n")], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "resultados_detector.csv" });
    a.click();
    URL.revokeObjectURL(url);
  }

  function abrirFila(i) {
    const f = lote.filas[i];
    const valores = {};
    Modelo.COLUMNAS_ENTRADA.forEach((c, j) => (valores[c] = lote.valores[i * 30 + j]));
    const sel = $("#ejemplo");
    let op = sel.querySelector('option[value="lote"]');
    if (!op) { op = document.createElement("option"); op.value = "lote"; sel.appendChild(op); }
    op.textContent = `Fila ${f.linea} del lote`;
    op._datos = { valores, clase: f.clase };
    sel.value = "lote";
    cargarEnFormulario(valores, f.clase);
    window.irA("tab-individual");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* ---------- transacción individual ---------- */
  function configurarFormulario() {
    const R = M.referencia;
    const sel = $("#ejemplo");
    const medianas = {};
    Modelo.COLUMNAS_ENTRADA.forEach((c) => (medianas[c] = c === "Time" ? 13 * 3600 : R.rangos[c].mediana));
    const opciones = [{ nombre: "Valores típicos (medianas)", valores: medianas, clase: null }].concat(R.ejemplos);
    opciones.forEach((o, i) => {
      const op = document.createElement("option");
      op.value = String(i); op.textContent = o.nombre; op._datos = o;
      sel.appendChild(op);
    });
    sel.addEventListener("change", () => {
      const d = sel.selectedOptions[0]._datos;
      cargarEnFormulario(d.valores, d.clase);
    });
    const hora = $("#f-hora");
    for (let h = 0; h < 24; h++) hora.insertAdjacentHTML("beforeend", `<option value="${h}">${String(h).padStart(2, "0")}:00</option>`);

    $("#deslizadores").innerHTML = DESLIZADORES.map((v) =>
      `<label class="deslizador"><span>${v}</span><input type="range" id="f-${v}" step="any"><output id="o-${v}"></output></label>`).join("");
    const otras = Array.from({ length: 28 }, (_, i) => "V" + (i + 1)).filter((v) => !DESLIZADORES.includes(v));
    $("#otras-v").innerHTML = otras.map((v) => `<label>${v}<input type="number" id="f-${v}" step="any"></label>`).join("");
    $("#formulario").addEventListener("input", leerFormulario);
    sel.value = "2";
    const d = opciones[2];
    cargarEnFormulario(d.valores, d.clase);
  }

  function cargarEnFormulario(valores, clase) {
    const R = M.referencia;
    $("#f-Amount").value = (+valores.Amount).toFixed(2);
    $("#f-hora").value = horaDe(valores.Time);
    for (let i = 1; i <= 28; i++) {
      const v = "V" + i, el = $("#f-" + v), x = +valores[v];
      if (el.type === "range") {
        const r = R.rangos[v];
        el.min = Math.min(r.p01, x).toFixed(3);
        el.max = Math.max(r.p99, x).toFixed(3);
      }
      el.value = x;
    }
    actual = { clase };
    leerFormulario();
  }

  function leerFormulario() {
    const fila = { Time: (+$("#f-hora").value) * 3600, Amount: Math.max(0, +$("#f-Amount").value || 0) };
    for (let i = 1; i <= 28; i++) fila["V" + i] = +$("#f-V" + i).value || 0;
    DESLIZADORES.forEach((v) => ($("#o-" + v).textContent = dec(2).format(fila[v])));
    actual = Object.assign(actual || {}, { fila });
    evaluarActual();
  }

  function evaluarActual() {
    const { fila, clase } = actual;
    if (!fila) return;
    const p = Modelo.puntuar(fila, M);
    const v = p[clave], u = umbral[clave], al = v >= u;
    $("#r-valor").textContent = fmtRiesgo(clave, v);
    $("#r-barra").style.left = (100 * escala.visual(clave, v)).toFixed(1) + "%";
    $("#r-marca").style.left = (100 * escala.visual(clave, u)).toFixed(1) + "%";
    const icono = al
      ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2 20h20z"/><path d="M12 10v4m0 3v.01"/></svg>'
      : '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>';
    const d = $("#r-decision");
    d.className = "decision " + (al ? "alerta" : "normal");
    d.innerHTML = icono + (al ? "Alerta: enviar a revisión" : "Normal: no requiere revisión");
    $("#r-contexto").textContent = `Puntaje de riesgo de 0 a 100 con ${NOMBRES[clave].toLowerCase()}. Esta transacción es más riesgosa que el ${dec(2).format(100 * percentil(clave, v))} % de las transacciones de validación; el umbral ${fmtUmbral(clave, u)}.`;
    const real = $("#r-real");
    real.hidden = clase === null || clase === undefined;
    if (!real.hidden) real.textContent = `Clase real de este ejemplo: ${clase === 1 ? "fraude" : "legítima"}. ` +
      ((clase === 1) === al ? "El modelo acierta." : clase === 1 ? "El modelo no lo detecta con este umbral." : "Es una falsa alarma con este umbral.");

    $("#r-modelos").innerHTML = Object.keys(NOMBRES).map((c) => {
      const supera = p[c] >= umbral[c];
      return `<div class="modelo-fila"><span>${NOMBRES[c]}</span>
        <div class="pista"><div style="width:${(100 * escala.visual(c, p[c])).toFixed(1)}%;background:${COLORES[c]}"></div></div>
        <output>${fmtRiesgo(c, p[c])}</output>
        <span class="chip ${supera ? "alerta" : "normal"}">${supera ? "Alerta" : "Normal"}</span></div>`;
    }).join("");

    // El puntaje del ensamble se satura cerca de 1, asi que su explicacion se arma
    // promediando los aportes en probabilidad de la red y de XGBoost.
    let aportes;
    if (clave === "ensamble") {
      const suma = {};
      for (const c of ["red", "xgboost"]) Modelo.explicar(fila, M, c).forEach((a) => (suma[a.variable] = (suma[a.variable] || 0) + a.aporte / 2));
      aportes = Object.entries(suma).map(([variable, aporte]) => ({ variable, aporte }))
        .sort((a, b) => Math.abs(b.aporte) - Math.abs(a.aporte));
    } else {
      aportes = Modelo.explicar(fila, M, clave);
    }
    aportes = aportes.slice(0, 6);
    $("#nota-explicacion").textContent = clave === "ensamble"
      ? "Para el ensamble se promedian los aportes de la red y de XGBoost. Los valores están en puntos del puntaje de 0 a 100."
      : "Los valores están en puntos del puntaje de 0 a 100.";
    const maximo = Math.max(1e-9, ...aportes.map((a) => Math.abs(a.aporte)));
    const unidad = 100;
    $("#r-explicacion").innerHTML = aportes.map((a) => {
      const w = (50 * Math.abs(a.aporte) / maximo).toFixed(1);
      const sube = a.aporte > 0;
      return `<div class="aporte"><span>${a.variable}</span><div class="eje"><div style="${sube ? "left:50%" : "right:50%"};width:${w}%;background:${sube ? "#D9534F" : "#12A37F"}"></div></div>
        <output>${sube ? "+" : "−"}${dec(1).format(Math.abs(a.aporte) * unidad)} pts</output></div>`;
    }).join("");
  }

  /* ---------- información ---------- */
  function llenarInfo() {
    const R = M.referencia;
    $("#info-filas").textContent = num.format(R.datos.prueba.filas);
    $("#info-fraudes").textContent = R.datos.prueba.fraudes;
    const filas = Object.keys(NOMBRES).map((c) => {
      const m = R.modelos[c].prueba;
      return `<tr><td>${NOMBRES[c]}</td><td class="num">${dec(3).format(m.auprc)}</td><td class="num">${dec(0).format(100 * m.recall)} %</td>
        <td class="num">${dec(0).format(100 * m.precision)} %</td><td class="num">${dec(3).format(m.f1)}</td>
        <td class="num">${m.vp} de ${m.vp + m.fn}</td><td class="num">${m.fp}</td><td class="num">${dec(1).format(m.monto_detectado_pct)} %</td></tr>`;
    }).join("");
    $("#tabla-metricas").innerHTML = `<thead><tr><th>Modelo</th><th class="num">AUPRC</th><th class="num">Recall</th><th class="num">Precisión</th>
      <th class="num">F1</th><th class="num">Fraudes detectados</th><th class="num">Alertas falsas</th><th class="num">Monto detectado</th></tr></thead><tbody>${filas}</tbody>`;
  }

  function enlaceRepositorio() {
    // En GitHub Pages la direccion es usuario.github.io/repositorio, de ahi se arma el enlace
    const a = $("#enlace-repo");
    const h = location.hostname;
    if (h.endsWith(".github.io")) {
      const repo = location.pathname.split("/").filter(Boolean)[0] || "";
      a.href = `https://github.com/${h.split(".")[0]}/${repo}`;
    } else {
      a.href = "https://github.com/";
    }
  }

  document.addEventListener("DOMContentLoaded", iniciar);
})();
