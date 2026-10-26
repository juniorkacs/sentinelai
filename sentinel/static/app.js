// SentinelAI dashboard. Everything shown comes from the API; nothing is computed here
// except slicing the chart window.
const $ = (id) => document.getElementById(id);
const NAMES = { robust_baseline: "Robust baseline", isolation_forest: "Isolation Forest" };
const METRIC_FIELDS = ["requests_per_s", "cpu_pct", "memory_pct", "p95_latency_ms", "error_rate_pct", "net_out_mb", "failed_logins"];

const state = { model: "robust_baseline", info: null, series: null, alerts: [], selected: null, window: null, chart: null };

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || (data.detail && JSON.stringify(data.detail)) || `Request failed (${res.status})`);
  return data;
}

const fmtTime = (iso) =>
  new Date(iso + "Z").toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

function renderKpis() {
  const r = state.info.models[state.model];
  const k = (label, value, sub) => el("div", { class: "kpi" }, el("p", { class: "l" }, label), el("p", { class: "v" }, value), el("p", { class: "s" }, sub));
  $("kpis").replaceChildren(
    k("Episodes detected", `${r.episodes_detected} of ${r.episodes}`, "labelled anomaly episodes in the test week"),
    k("Precision", r.precision.toFixed(2), "share of flagged minutes that were anomalous"),
    k("Recall", r.recall.toFixed(2), "share of anomalous minutes flagged"),
    k("False alerts per day", r.false_alerts_per_day, `${r.false_alerts} of ${r.alerts} alerts`),
    k("Average precision", r.average_precision.toFixed(2), "ranking quality across all thresholds"),
  );
}

function windowSlice(len) {
  if (!state.window) return [0, len];
  return [Math.max(0, state.window[0]), Math.min(len, state.window[1])];
}

function renderChart() {
  const s = state.series;
  const [a, b] = windowSlice(s.timestamps.length);
  const labels = s.timestamps.slice(a, b);
  const values = s.values.slice(a, b);
  const flags = s.flags[state.model].slice(a, b);
  const truth = s.labelled_anomaly.slice(a, b);
  const max = Math.max(...values, ...(s.band_high ? s.band_high.slice(a, b) : []));
  const data = {
    labels,
    datasets: [
      { label: "Typical high", data: s.band_high.slice(a, b), borderColor: "rgba(45,82,128,0.9)", borderWidth: 1, pointRadius: 0, fill: false },
      { label: "Typical low", data: s.band_low.slice(a, b), borderColor: "rgba(45,82,128,0.9)", borderWidth: 1, pointRadius: 0, fill: "-1", backgroundColor: "rgba(45,82,128,0.25)" },
      { label: s.label, data: values, borderColor: "#45c8e6", borderWidth: 1.2, pointRadius: 0 },
      { label: "Flagged", data: values.map((v, i) => (flags[i] ? v : null)), showLine: false, pointRadius: 2.5, pointBackgroundColor: "#b39dff", pointBorderWidth: 0 },
      { label: "Labelled anomaly", data: truth.map((t) => (t ? max * 1.04 : null)), borderColor: "rgba(240,181,74,0.9)", borderWidth: 4, pointRadius: 0, spanGaps: false },
    ],
  };
  const opts = {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        filter: (item) => item.datasetIndex === 2,
        callbacks: { title: (items) => fmtTime(items[0].label) },
      },
    },
    scales: {
      x: {
        ticks: { color: "#93a7bf", maxTicksLimit: 8, callback: (v) => fmtTime(labels[v]) },
        grid: { color: "#1f3a5c" },
      },
      y: { ticks: { color: "#93a7bf" }, grid: { color: "#1f3a5c" }, beginAtZero: true },
    },
  };
  $("chart-title").textContent = state.window ? `${s.label}, around the selected alert` : `${s.label}, test week`;
  $("full-week").hidden = !state.window;
  if (state.chart) state.chart.destroy();
  state.chart = new window.Chart($("chart"), { type: "line", data, options: opts });
}

async function loadSeries(metric) {
  state.series = await getJSON(`/api/series?metric=${encodeURIComponent(metric)}`);
  $("metric").value = metric;
  renderChart();
}

function renderAlerts() {
  const hide = $("hide-false").checked;
  const list = state.alerts.filter((a) => !hide || !a.ground_truth.startsWith("no "));
  const falseCount = state.alerts.filter((a) => a.ground_truth.startsWith("no ")).length;
  $("alert-count").textContent = `${state.alerts.length} alerts, ${falseCount} false`;
  $("alerts").replaceChildren(
    ...list.map((a) => {
      const isFalse = a.ground_truth.startsWith("no ");
      const top = a.explanation[0];
      return el(
        "li",
        {},
        el(
          "button",
          { type: "button", "aria-current": String(state.selected === a.id), onclick: () => select(a.id) },
          el("span", { class: "when" }, `${fmtTime(a.start)}, ${a.minutes} min`, el("span", { class: `tag ${isFalse ? "bad" : "ok"}` }, isFalse ? "false alert" : a.ground_truth)),
          el("span", { class: "why" }, top ? top.summary : ""),
        ),
      );
    }),
  );
}

function select(id) {
  state.selected = id;
  const a = state.alerts.find((x) => x.id === id);
  renderAlerts();
  const body = $("detail-body");
  const factors = el("ol", { class: "factors" });
  for (const f of a.explanation) {
    const bar = el("span");
    bar.style.width = `${Math.round(f.share * 100)}%`;
    factors.append(
      el(
        "li",
        { class: "factor" },
        el("div", { class: "name" }, el("span", {}, f.label), el("span", { class: "muted" }, `${Math.round(f.share * 100)}% of the anomaly`)),
        el("div", { class: "bar", "aria-hidden": "true" }, bar),
        el("p", { class: "vals" }, `${f.summary}: ${f.value} against a typical ${f.typical} (z = ${f.z}).`),
      ),
    );
  }
  const isFalse = a.ground_truth.startsWith("no ");
  body.replaceChildren(
    el("p", {}, `${NAMES[state.model]} scored this ${a.peak_score} at ${fmtTime(a.peak_time)}, against a threshold of ${a.threshold}. It lasted ${a.minutes} minute${a.minutes === 1 ? "" : "s"}.`),
    factors,
    el("p", { class: "truth" },
      isFalse
        ? "Ground truth: no anomaly was injected here, so this is a false alert. In real operations you would not know this without investigating."
        : `Ground truth: ${a.ground_truth}. ${a.ground_truth_description}`),
    el("p", { class: "muted small" }, state.info.disclaimer),
  );
  // Show the top factor's raw metric around the alert.
  const top = a.explanation.find((f) => METRIC_FIELDS.includes(f.feature)) || a.explanation[0];
  const metric = METRIC_FIELDS.includes(top.feature) ? top.feature : "requests_per_s";
  const idx = state.series.timestamps.indexOf(a.start);
  state.window = idx >= 0 ? [idx - 240, idx + a.minutes + 240] : null;
  loadSeries(metric);
}

async function loadModel(model) {
  state.model = model;
  state.selected = null;
  document.querySelectorAll(".model-pick button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.model === model)));
  renderKpis();
  const res = await getJSON(`/api/alerts?model=${model}`);
  state.alerts = res.alerts;
  renderAlerts();
  $("detail-body").replaceChildren(el("p", { class: "muted" }, "Select an alert to see which signals made it unusual."));
  if (state.series) renderChart();
}

// ----- custom observation scoring -----
function buildScoreForm() {
  const fields = $("score-fields");
  const first = state.info.dataset.test_start;
  fields.replaceChildren(
    el("div", {}, el("label", { for: "f-time" }, "Time (UTC)"), el("input", { id: "f-time", type: "datetime-local", value: first.slice(0, 16) })),
    ...METRIC_FIELDS.map((m) => {
      const label = state.info.dataset.features.find((f) => f.name === m).label;
      return el("div", {}, el("label", { for: `f-${m}` }, label), el("input", { id: `f-${m}`, type: "number", step: "any", min: "0", required: "" }));
    }),
  );
}

async function fillTypical() {
  // Use the median of the first test hour as a realistic starting point.
  const vals = {};
  for (const m of METRIC_FIELDS) {
    const s = await getJSON(`/api/series?metric=${m}`);
    const hour = s.values.slice(0, 60).sort((a, b) => a - b);
    vals[m] = hour[30];
  }
  for (const m of METRIC_FIELDS) $(`f-${m}`).value = vals[m];
  $("score-result").replaceChildren();
}

async function submitScore(e) {
  e.preventDefault();
  const obs = { timestamp: new Date($("f-time").value + ":00Z").toISOString() };
  for (const m of METRIC_FIELDS) obs[m] = Number($(`f-${m}`).value);
  const out = $("score-result");
  try {
    const res = await getJSON("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: state.model, observations: [obs] }),
    });
    const r = res.results[0];
    out.replaceChildren(
      el("p", { class: "big" }, r.is_anomalous ? "Unusual" : "Within normal range"),
      el("p", { class: "muted small" }, `${NAMES[res.model]} score ${r.score} (threshold ${res.threshold}).`),
      el("ul", {}, ...r.top_factors.map((f) => el("li", { class: "small" }, `${f.summary} (${Math.round(f.share * 100)}%).`))),
    );
  } catch (err) {
    out.replaceChildren(el("p", { class: "error" }, err.message));
  }
}

async function init() {
  try {
    state.info = await getJSON("/api/model");
    $("disclaimer").textContent = state.info.disclaimer;
    $("metric").replaceChildren(...state.info.dataset.metrics.map((m) => el("option", { value: m }, state.info.dataset.features.find((f) => f.name === m).label)));
    $("metric").addEventListener("change", () => loadSeries($("metric").value));
    $("full-week").addEventListener("click", () => { state.window = null; renderChart(); });
    $("hide-false").addEventListener("change", renderAlerts);
    document.querySelectorAll(".model-pick button").forEach((b) => b.addEventListener("click", () => loadModel(b.dataset.model)));
    buildScoreForm();
    $("score-form").addEventListener("submit", submitScore);
    $("reset-typical").addEventListener("click", fillTypical);
    await loadSeries("cpu_pct");
    await loadModel("robust_baseline");
    await fillTypical();
  } catch (err) {
    $("kpis").replaceChildren(el("p", { class: "error" }, `Could not load data from the API: ${err.message}`));
  }
}

init();
