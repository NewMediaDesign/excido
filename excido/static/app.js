// Excido — UI logic.
import { applyI18n, getLang, has, MOD, setLang, t, tn } from "./i18n.js";
import { Viewer } from "./viewer.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// ───────────────────────── state ─────────────────────────

const DEFAULT_INFER = {
  model: "general", resolution: "auto", width: 1024, height: 1024, max_side: 2048,
  device: "auto", precision: "fp16", matmul_precision: "high", keep_loaded: true,
  oom_fallback: "error", mask_interp: "bilinear", batch_size: 1,
};
const DEFAULT_POST = {
  mask_mode: "soft", threshold: 0.5, invert: false, grow: 0, refine: true, refine_r1: 90, refine_r2: 6,
  background: "transparent", bg_color: "#ffffff", bg_blur: 20, export_format: "png", bg_image_id: null,
};
const STORE_KEY = "excido.params.v1";

const state = {
  infer: { ...DEFAULT_INFER },
  post: { ...DEFAULT_POST },
  bg: null, // { id, url, name }
  system: null,
  models: [],
  examples: [],
  history: [],
  current: null, // description of the run on screen
  view: "single",
  busy: false,
  batch: { items: [], job: null, seq: 0, lastJob: null },
  downloads: {}, // model_id -> job
};

function loadSaved() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (saved?.infer) Object.assign(state.infer, pick(saved.infer, DEFAULT_INFER));
    if (saved?.post) Object.assign(state.post, pick(saved.post, DEFAULT_POST));
    if (saved?.bg) state.bg = saved.bg;
  } catch { /* storage unavailable: defaults */ }
}
function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ infer: state.infer, post: state.post, bg: state.bg })); } catch {}
}
function pick(obj, ref) {
  return Object.fromEntries(Object.keys(ref).filter((k) => k in obj).map((k) => [k, obj[k]]));
}

// ───────────────────────── API ─────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch {}
  if (!res.ok) {
    const err = new Error(data?.error || (typeof data?.detail === "string" ? data.detail : `HTTP ${res.status}`));
    err.code = data?.code;
    err.params = data?.params || {};
    err.hints = data?.hints || [];
    throw err;
  }
  return data;
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ───────────────────────── toasts ─────────────────────────

function toast(title, { body = "", hints = [], kind = "info", ms = 5000 } = {}) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = "<strong></strong><span></span>";
  el.querySelector("strong").textContent = title;
  el.querySelector("span").textContent = body;
  if (hints.length) {
    const ul = document.createElement("ul");
    for (const h of hints) { const li = document.createElement("li"); li.textContent = h; ul.append(li); }
    el.append(ul);
  }
  $("#toasts").append(el);
  setTimeout(() => el.remove(), hints.length ? ms * 2 : ms);
}
function errorText(err) {
  return err.code ? t(`err.${err.code}`, err.params, err.message) : err.message || String(err);
}
const toastError = (err) => toast(errorText(err), { kind: "error", hints: (err.hints || []).map((h) => t(`hint.${h}`)) });

// ───────────────────────── formatting ─────────────────────────

const fmtMs = (ms) => (ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const fmtMB = (bytes) => `${Math.round(bytes / 1e6)} MB`;
const fmtGB = (mb) => (mb == null ? "—" : mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb} MB`);
const fmtRes = (r) => (r ? `${r[0]}×${r[1]}` : t("opt.dynamic"));
const modelById = (id) => state.models.find((m) => m.id === id);
const modelText = (m, field) => (getLang() !== "en" && has(`model.${m.id}.${field}`) ? t(`model.${m.id}.${field}`) : m[field]);
const deviceLabel = (id) => t(`dev.${id}`);
const effectiveDevice = () => (state.infer.device === "auto" || !state.system?.devices.some((d) => d.id === state.infer.device)
  ? state.system?.default_device || "cpu"
  : state.infer.device);

// ───────────────────────── viewer ─────────────────────────

const viewer = new Viewer($("#stage"), $("#frame"), {
  onZoom: (s) => { $("#zoom-value").textContent = `${Math.round(s * 100)}%`; },
});

// ───────────────────────── controls ─────────────────────────

function bindSeg(id, target, key, onChange) {
  $(`#${id}`).addEventListener("click", (e) => {
    const b = e.target.closest("button[data-value]");
    if (!b || b.disabled) return;
    target[key] = b.dataset.value;
    syncSeg(id, target[key]);
    onChange(key);
  });
}
function syncSeg(id, value) {
  for (const b of $$(`#${id} button[data-value]`)) b.setAttribute("aria-checked", String(b.dataset.value === value));
}
function bindRange(id, target, key, onChange, { fmt = (v) => v } = {}) {
  const input = $(`#${id}`);
  const out = $(`#o-${key}`);
  input.addEventListener("input", () => {
    target[key] = Number(input.value);
    if (out) out.textContent = fmt(target[key]);
    onChange(key);
  });
}
function syncRange(id, key, value, fmt = (v) => v) {
  $(`#${id}`).value = value;
  const out = $(`#o-${key}`);
  if (out) out.textContent = fmt(value);
}

const onInfer = () => { save(); syncResolutionUI(); updateRunButton(); };
const onPost = () => { save(); syncPostUI(); schedulePostprocess(); };

function bindControls() {
  $("#p-model").addEventListener("change", (e) => { state.infer.model = e.target.value; renderModelCard(); onInfer(); });

  bindSeg("p-resolution", state.infer, "resolution", onInfer);
  for (const k of ["width", "height"]) {
    $(`#p-${k}`).addEventListener("change", (e) => {
      state.infer[k] = snap32(Number(e.target.value) || 1024);
      e.target.value = state.infer[k];
      onInfer();
    });
  }
  $("#res-presets").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    Object.assign(state.infer, { resolution: "custom", width: Number(b.dataset.w), height: Number(b.dataset.h) });
    syncInferUI();
    onInfer();
  });
  bindRange("p-max_side", state.infer, "max_side", onInfer, { fmt: (v) => `${v} px` });

  bindSeg("p-device", state.infer, "device", onInfer);
  bindSeg("p-precision", state.infer, "precision", onInfer);
  $("#p-matmul_precision").addEventListener("change", (e) => { state.infer.matmul_precision = e.target.value; onInfer(); });
  bindRange("p-batch_size", state.infer, "batch_size", onInfer);
  $("#p-oom_cpu").addEventListener("change", (e) => { state.infer.oom_fallback = e.target.checked ? "cpu" : "error"; onInfer(); });
  $("#p-keep_loaded").addEventListener("change", (e) => { state.infer.keep_loaded = e.target.checked; onInfer(); });
  $("#p-mask_interp").addEventListener("change", (e) => { state.infer.mask_interp = e.target.value; onInfer(); });

  bindSeg("q-mask_mode", state.post, "mask_mode", onPost);
  bindRange("q-threshold", state.post, "threshold", onPost, { fmt: (v) => v.toFixed(2) });
  $("#q-invert").addEventListener("change", (e) => { state.post.invert = e.target.checked; onPost(); });

  // Grow / shrink: slider for quick moves, stepper and number field for half-pixel precision.
  const setGrow = (v) => {
    const step = state.post.mask_mode === "binary" ? 1 : 0.5;
    v = Math.round((Number(v) || 0) / step) * step;
    state.post.grow = Math.min(100, Math.max(-100, v));
    onPost();
  };
  $("#q-grow").addEventListener("input", (e) => setGrow(e.target.value));
  $("#q-grow").addEventListener("dblclick", () => setGrow(0));
  $("#q-grow-num").addEventListener("change", (e) => setGrow(String(e.target.value).replace(",", ".")));
  $("#q-grow-num").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    const step = Number($("#q-grow").step) * (e.shiftKey ? 10 : 1);
    setGrow(state.post.grow + (e.key === "ArrowUp" ? step : -step));
  });
  $("#grow-minus").addEventListener("click", () => setGrow(state.post.grow - Number($("#q-grow").step)));
  $("#grow-plus").addEventListener("click", () => setGrow(state.post.grow + Number($("#q-grow").step)));
  $("#q-refine").addEventListener("change", (e) => { state.post.refine = e.target.checked; onPost(); });
  bindRange("q-refine_r1", state.post, "refine_r1", onPost);
  bindRange("q-refine_r2", state.post, "refine_r2", onPost);
  bindSeg("q-background", state.post, "background", onPost);
  $("#swatches").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-color]");
    if (!b) return;
    state.post.bg_color = b.dataset.color;
    onPost();
  });
  $("#q-bg_color").addEventListener("input", (e) => { state.post.bg_color = e.target.value; onPost(); });
  bindRange("q-bg_blur", state.post, "bg_blur", onPost, { fmt: (v) => `${v} px` });
  $("#bg-input").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      state.bg = await api("/api/backgrounds", { method: "POST", body: fd });
      state.post.bg_image_id = state.bg.id;
      state.post.background = "image";
      onPost();
    } catch (err) { toastError(err); }
  });
  bindSeg("q-export_format", state.post, "export_format", onPost);

  $("#reset-params").addEventListener("click", () => {
    Object.assign(state.infer, DEFAULT_INFER);
    Object.assign(state.post, DEFAULT_POST);
    state.bg = null;
    syncAllUI();
    renderModelCard();
    save();
    updateRunButton();
    schedulePostprocess();
    toast(t("toast.reset"), { kind: "ok", ms: 2500 });
  });
}

function snap32(v) {
  return Math.min(4096, Math.max(256, Math.floor(v / 32) * 32));
}

function renderDeviceButtons() {
  const root = $("#p-device");
  const devices = state.system?.devices || [{ id: "cpu", name: "CPU" }];
  root.replaceChildren();
  const auto = document.createElement("button");
  auto.setAttribute("role", "radio");
  auto.dataset.value = "auto";
  auto.textContent = t("dev.auto");
  auto.title = t("dev.auto.title", { device: deviceLabel(state.system?.default_device || "cpu") });
  root.append(auto);
  for (const d of devices) {
    const b = document.createElement("button");
    b.setAttribute("role", "radio");
    b.dataset.value = d.id;
    b.textContent = deviceLabel(d.id);
    b.title = d.name;
    root.append(b);
  }
  if (state.system && state.infer.device !== "auto" && !devices.some((d) => d.id === state.infer.device)) state.infer.device = "auto";
  syncSeg("p-device", state.infer.device);
}

function syncInferUI() {
  const p = state.infer;
  $("#p-model").value = p.model;
  syncSeg("p-resolution", p.resolution);
  $("#p-width").value = p.width;
  $("#p-height").value = p.height;
  syncRange("p-max_side", "max_side", p.max_side, (v) => `${v} px`);
  syncSeg("p-device", p.device);
  $("#p-matmul_precision").value = p.matmul_precision;
  syncRange("p-batch_size", "batch_size", p.batch_size);
  $("#p-oom_cpu").checked = p.oom_fallback === "cpu";
  $("#p-keep_loaded").checked = p.keep_loaded;
  $("#p-mask_interp").value = p.mask_interp;
  syncResolutionUI();
}

function syncPostUI() {
  const q = state.post;
  syncSeg("q-mask_mode", q.mask_mode);
  syncRange("q-threshold", "threshold", q.threshold, (v) => Number(v).toFixed(2));
  $("#threshold-field").style.opacity = q.mask_mode === "binary" ? "1" : "0.45";
  $("#q-threshold").disabled = q.mask_mode !== "binary";
  $("#q-invert").checked = q.invert;
  const step = q.mask_mode === "binary" ? 1 : 0.5; // binary masks move by whole pixels
  if (step === 1 && !Number.isInteger(q.grow)) q.grow = Math.round(q.grow);
  $("#q-grow").step = String(step);
  $("#q-grow").value = Math.min(25, Math.max(-25, q.grow));
  $("#q-grow-num").value = (q.grow > 0 ? "+" : "") + q.grow.toFixed(1);
  $("#grow-field").classList.toggle("grow-active", q.grow !== 0);
  $("#q-refine").checked = q.refine;
  syncRange("q-refine_r1", "refine_r1", q.refine_r1);
  syncRange("q-refine_r2", "refine_r2", q.refine_r2);
  $("#q-refine_r1").disabled = $("#q-refine_r2").disabled = !q.refine;
  syncSeg("q-background", q.background);
  $("#bg-color-opt").hidden = q.background !== "color";
  $("#bg-blur-opt").hidden = q.background !== "blur";
  $("#bg-image-opt").hidden = q.background !== "image";
  $("#q-bg_color").value = q.bg_color;
  for (const b of $$("#swatches button")) b.classList.toggle("on", b.dataset.color.toLowerCase() === q.bg_color.toLowerCase());
  syncRange("q-bg_blur", "bg_blur", q.bg_blur, (v) => `${v} px`);
  const prev = $("#bg-preview");
  prev.hidden = !state.bg;
  if (state.bg) prev.style.backgroundImage = `url("${state.bg.url}")`;
  syncSeg("q-export_format", q.export_format);
}

function syncAllUI() {
  syncInferUI();
  syncPostUI();
}

function syncResolutionUI() {
  const p = state.infer;
  const m = modelById(p.model);
  const dynamic = m?.dynamic;
  $("#res-custom").hidden = p.resolution !== "custom";
  $("#res-maxside").hidden = !(p.resolution === "original" || (p.resolution === "auto" && dynamic));
  for (const b of $$("#res-presets button")) {
    b.classList.toggle("on", p.resolution === "custom" && Number(b.dataset.w) === p.width && Number(b.dataset.h) === p.height);
  }
  const res = fmtRes(m?.native);
  let hint;
  if (p.resolution === "auto") hint = dynamic ? t("res.hint.dynamic", { max: p.max_side }) : t("res.hint.native", { res });
  else if (p.resolution === "custom") hint = t("res.hint.custom", { w: p.width, h: p.height, res });
  else hint = t("res.hint.original", { max: p.max_side });
  $("#res-hint").innerHTML = hint;

  // Precision availability depends on the device actually used.
  const dev = effectiveDevice();
  const bf16 = dev === "cuda" && state.system?.bf16_supported;
  for (const b of $$("#p-precision button")) {
    b.disabled = (dev === "cpu" && b.dataset.value !== "fp32") || (b.dataset.value === "bf16" && !bf16);
  }
  const shown = dev === "cpu" ? "fp32" : p.precision === "bf16" && !bf16 ? "fp16" : p.precision;
  syncSeg("p-precision", shown);
}

// ───────────────────────── models ─────────────────────────

async function refreshModels() {
  const data = await api("/api/models");
  state.models = data.models;
  for (const m of state.models) {
    if (m.download_job && !state.downloads[m.id]) watchDownload({ id: m.download_job, model: m.id, progress: 0 });
  }
  renderModelSelect();
  renderModelCard();
  syncResolutionUI();
  updateRunButton();
}

function renderModelSelect() {
  const sel = $("#p-model");
  const groups = new Map();
  for (const m of state.models) {
    if (!groups.has(m.group)) groups.set(m.group, []);
    groups.get(m.group).push(m);
  }
  sel.replaceChildren();
  for (const [group, list] of groups) {
    const og = document.createElement("optgroup");
    og.label = t(`group.${group}`);
    for (const m of list) {
      const o = document.createElement("option");
      o.value = m.id;
      const extra = [m.dynamic ? t("opt.dynamic") : fmtRes(m.native)];
      if (!m.downloaded) extra.push(t("opt.missing"));
      if (m.loaded) extra.push(t("opt.loaded"));
      o.textContent = `${m.label} · ${extra.join(" · ")}`;
      og.append(o);
    }
    sel.append(og);
  }
  sel.value = state.infer.model;
  if (!sel.value && state.models.length) { state.infer.model = state.models[0].id; sel.value = state.infer.model; }
}

function renderModelCard() {
  const m = modelById(state.infer.model);
  const card = $("#model-card");
  if (!m) { card.textContent = "—"; return; }
  const job = state.downloads[m.id];
  const status = m.loaded
    ? `<span class="pill loaded">${t("pill.loaded")}</span>`
    : m.downloaded ? `<span class="pill ok">${t("pill.ok")}</span>` : `<span class="pill missing">${t("pill.missing")}</span>`;
  const license = m.license === "MIT" ? "MIT" : t("mc.noLicense");
  card.innerHTML = `
    <div class="mc-task"></div>
    <div class="mc-notes"></div>
    <dl>
      <dt>${t("mc.backbone")}</dt><dd>${m.backbone === "swin_v1_l" ? "Swin-L · 220.7 M" : "Swin-T · 44.4 M"}</dd>
      <dt>${t("mc.resolution")}</dt><dd>${m.dynamic ? t("mc.dynamic") : fmtRes(m.native)}</dd>
      <dt>${t("mc.weights")}</dt><dd>${fmtMB(m.size_bytes)} · ${m.weights_dtype.toUpperCase()}</dd>
      <dt>${t("mc.license")}</dt><dd class="${m.license === "MIT" ? "" : "warn"}">${license}</dd>
      <dt>${t("mc.repo")}</dt><dd><a href="https://huggingface.co/${m.repo}" target="_blank" rel="noopener">${m.repo}</a></dd>
      <dt>${t("mc.revision")}</dt><dd>${m.revision.slice(0, 10)}</dd>
    </dl>
    <div class="mc-actions">${status}<span class="mc-dl"></span></div>`;
  card.querySelector(".mc-task").textContent = modelText(m, "task");
  card.querySelector(".mc-notes").textContent = modelText(m, "notes");
  const slot = card.querySelector(".mc-dl");
  if (job) {
    slot.style.cssText = "display:flex;flex:1;align-items:center;gap:8px";
    const pct = Math.round((job.progress || 0) * 100);
    slot.innerHTML = `<div class="mini-progress ${pct < 1 ? "indeterminate" : ""}"><div style="width:${pct}%"></div></div><span class="hint">${pct}%</span>`;
  } else if (!m.downloaded) {
    const b = document.createElement("button");
    b.className = "btn btn-primary";
    b.textContent = t("mc.download", { size: fmtMB(m.size_bytes) });
    b.addEventListener("click", () => startDownload(m.id));
    slot.append(b);
  } else if (m.loaded) {
    const b = document.createElement("button");
    b.className = "btn btn-quiet";
    b.textContent = t("mc.unload");
    b.addEventListener("click", async () => {
      try { await api("/api/models/unload", { method: "POST" }); await Promise.all([refreshModels(), refreshSystem()]); }
      catch (err) { toastError(err); }
    });
    slot.append(b);
  }
}

async function startDownload(id) {
  try {
    watchDownload(await api(`/api/models/${id}/download`, { method: "POST" }));
  } catch (err) { toastError(err); }
}

function watchDownload(job) {
  state.downloads[job.model] = job;
  renderModelCard();
  const label = modelById(job.model)?.label || job.model;
  const timer = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${job.id}`);
      state.downloads[job.model] = j;
      if (j.status === "done" || j.status === "error") {
        clearInterval(timer);
        delete state.downloads[job.model];
        if (j.status === "done") toast(t("toast.downloaded", { label }), { kind: "ok" });
        else toast(t("toast.downloadFailed", { label }), { kind: "error", body: j.message });
        await refreshModels();
      } else if (state.infer.model === job.model) {
        renderModelCard();
      }
    } catch (err) { clearInterval(timer); delete state.downloads[job.model]; toastError(err); }
  }, 700);
}

// ───────────────────────── system ─────────────────────────

async function refreshSystem() {
  try { state.system = await api("/api/system"); } catch { return; }
  const s = state.system;
  const main = s.devices[0];
  const gpu = main.id !== "cpu";
  const dev = $("#chip-device");
  dev.className = `chip ${gpu ? "ok" : "warn"}`;
  // Short name in the chip ("RTX 4070 Laptop"), full name in the tooltip.
  const shortName = main.name.replace(/^NVIDIA\s+(GeForce\s+)?/i, "").replace(/\s+GPU$/i, "");
  dev.querySelector("span").textContent = gpu
    ? `${shortName}${main.id === "cuda" && s.cuda_version ? ` · CUDA ${s.cuda_version}` : ""}`
    : `${t("chip.cpuOnly")} · ${main.name}`;
  dev.title = `${t("chip.device.title")}: ${main.name}`;
  $("#chip-vram span").textContent = s.memory_total_mb ? `${gpu ? deviceLabel(main.id) : "RAM"} ${fmtGB(s.memory_used_mb)} / ${fmtGB(s.memory_total_mb)}` : `torch ${s.torch}`;
  const mc = $("#chip-model");
  if (s.loaded) {
    const m = modelById(s.loaded.model);
    mc.classList.add("active");
    mc.querySelector("span").textContent = `${m?.label || s.loaded.model} · ${deviceLabel(s.loaded.device)} · ${s.loaded.dtype}`;
  } else {
    mc.classList.remove("active");
    mc.querySelector("span").textContent = t("chip.model.none");
  }
  const devButtons = $$("#p-device button").map((b) => b.dataset.value).join();
  if (devButtons !== ["auto", ...s.devices.map((d) => d.id)].join()) renderDeviceButtons();
}

// ───────────────────────── examples and history ─────────────────────────

async function refreshExamples() {
  state.examples = (await api("/api/examples")).groups;
  renderExamples();
}

function renderExamples() {
  const root = $("#examples");
  root.replaceChildren();
  for (const g of state.examples) {
    if (!g.items.length) continue;
    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="example-group-title"></div><div class="thumb-grid"></div>`;
    wrap.querySelector(".example-group-title").textContent = g.id === "official" ? t("examples.official") : t("examples.gallery", { folder: g.folder });
    const grid = wrap.querySelector(".thumb-grid");
    for (const it of g.items) {
      const b = document.createElement("button");
      b.className = "thumb checker";
      b.dataset.source = it.source;
      b.title = it.hint ? `${it.name}\n${t("examples.hint", { model: modelById(it.hint.model)?.label || it.hint.model })}` : it.name;
      b.innerHTML = `<img loading="lazy" alt=""><span class="thumb-label"></span>`;
      b.querySelector("img").src = it.url;
      b.querySelector(".thumb-label").textContent = it.name;
      b.addEventListener("click", () => {
        if (state.view === "batch") addBatchItems([{ name: it.name, source: it.source, thumb: it.url }]);
        else runSingle({ source: it.source }, it.name);
      });
      grid.append(b);
    }
    root.append(wrap);
  }
  if (!root.children.length) root.innerHTML = `<p class="hint">${t("examples.none")}</p>`;
}

async function refreshHistory() {
  state.history = (await api("/api/runs?limit=40")).runs;
  renderHistory();
}

function renderHistory() {
  const root = $("#history");
  if (!state.history.length) {
    root.innerHTML = `<p class="empty">${t("history.empty")}</p>`;
    return;
  }
  root.replaceChildren();
  for (const r of state.history) {
    const b = document.createElement("button");
    b.className = "history-item" + (state.current?.id === r.id ? " current" : "");
    const info = r.info || {};
    const size = info.input_sizes?.[0];
    b.innerHTML = `<span class="hthumb checker"><img alt="" loading="lazy"></span><span style="min-width:0"><div class="hname"></div><div class="hmeta"></div></span>`;
    b.querySelector("img").src = r.urls.thumb;
    b.querySelector(".hname").textContent = r.name;
    b.querySelector(".hmeta").textContent = `${info.model_label || info.model} · ${size ? size.join("×") : ""} · ${fmtMs(info.timings_ms?.inference)}`;
    b.addEventListener("click", () => openRun(r));
    root.append(b);
  }
}

function openRun(desc, { restorePost = true } = {}) {
  if (restorePost && desc.post) {
    Object.assign(state.post, pick(desc.post, DEFAULT_POST));
    syncPostUI();
    save();
  }
  showRun(desc);
  if (state.view !== "single") switchView("single");
}

// ───────────────────────── single run ─────────────────────────

function setLoading(on, text = t("loading.default")) {
  state.busy = on;
  $("#stage-loading").hidden = !on;
  $("#stage-empty").hidden = on || !!state.current;
  $("#loading-text").textContent = text;
  updateRunButton();
}

async function runSingle(input, label = "") {
  if (state.busy) return;
  const m = modelById(state.infer.model);
  if (m && !m.downloaded) {
    toast(t("err.model_not_downloaded", { label: m.label }), { kind: "error", hints: [t("hint.download_model")] });
    return;
  }
  const fd = new FormData();
  if (input.file) fd.append("file", input.file, input.file.name || "pasted.png");
  else if (input.url) fd.append("url", input.url);
  else if (input.source) fd.append("source", input.source);
  fd.append("params", JSON.stringify(state.infer));
  fd.append("post", JSON.stringify(state.post));
  const loaded = state.system?.loaded?.model === state.infer.model;
  setLoading(true, loaded ? t("loading.segment", { name: label }) : t("loading.load", { model: m?.label || state.infer.model }));
  try {
    const desc = await api("/api/segment", { method: "POST", body: fd });
    showRun(desc);
    state.history = [desc, ...state.history.filter((r) => r.id !== desc.id)];
    renderHistory();
    if (desc.info?.fallback) toast(t("m.fallback"));
  } catch (err) {
    toastError(err);
  } finally {
    setLoading(false);
    refreshSystem();
    refreshModels().catch(() => {});
  }
}

function showRun(desc) {
  state.current = desc;
  $("#stage-empty").hidden = true;
  viewer.setData({
    original: desc.urls.original, cutout: desc.urls.cutout, mask: desc.urls.mask, overlay: desc.urls.overlay,
    width: desc.width, height: desc.height,
  });
  updateDownloads();
  renderMetrics();
  renderHistory();
  updateRunButton();
}

function updateDownloads() {
  const d = state.current;
  const cut = $("#dl-cutout");
  const mask = $("#dl-mask");
  const psd = $("#dl-psd");
  if (!d) {
    for (const a of [cut, mask, psd]) a.setAttribute("aria-disabled", "true");
    return;
  }
  // Built on demand by the server with the settings currently shown.
  psd.href = `/api/runs/${d.id}/psd?lang=${getLang()}&v=${Math.round((d.updated || 0) * 1000)}`;
  psd.download = `${d.name}.psd`;
  psd.removeAttribute("aria-disabled");
  const ext = d.cutout_file.split(".").pop();
  cut.href = d.urls.cutout;
  cut.download = `${d.name}_${t(d.post?.background === "transparent" ? "file.cutout" : "file.composite")}.${ext}`;
  mask.href = d.urls.mask;
  mask.download = `${d.name}_${t("file.mask")}.png`;
  cut.removeAttribute("aria-disabled");
  mask.removeAttribute("aria-disabled");
}

function renderMetrics() {
  const d = state.current;
  const root = $("#metrics");
  if (!d) return;
  const i = d.info || {};
  const tm = i.timings_ms || {};
  const p = d.post_timings_ms || {};
  const items = [
    [t("m.original"), `${d.width}×${d.height}`],
    [t("m.input"), (i.input_sizes?.[0] || []).join("×")],
    [t("m.model"), i.model_label || i.model],
    [t("m.compute"), `${deviceLabel(i.device || "cpu")} · ${(i.precision || "").toUpperCase()}`],
  ];
  if (tm.load) items.push([t("m.load"), fmtMs(tm.load)]);
  items.push([t("m.preprocess"), fmtMs(tm.preprocess)], [t("m.inference"), fmtMs(tm.inference)], [t("m.resize"), fmtMs(tm.resize)]);
  if (d.post?.refine) items.push([t("m.refine"), fmtMs(p.refine)]);
  items.push([t("m.compose"), fmtMs((p.compose || 0) + (p.mask || 0))], [t("m.save"), fmtMs(p.save)]);
  const peak = i.memory_peak_mb ?? i.vram_peak_mb;
  if (peak != null) items.push([t("m.memory"), fmtGB(peak)]);
  root.replaceChildren();
  for (const [k, v] of items) {
    const s = document.createElement("span");
    s.className = "metric";
    s.innerHTML = "<b></b><span></span>";
    s.querySelector("b").textContent = k;
    s.querySelector("span").textContent = v;
    root.append(s);
  }
  if (i.fallback) {
    const s = document.createElement("span");
    s.className = "metric warn";
    s.textContent = t("m.fallback");
    root.append(s);
  }
}

// ───────────────────────── live post-processing ─────────────────────────

let postTimer = null;
let postSeq = 0;
function schedulePostprocess() {
  if (!state.current) return;
  clearTimeout(postTimer);
  postTimer = setTimeout(runPostprocess, 180);
}
async function runPostprocess() {
  const d = state.current;
  if (!d) return;
  const seq = ++postSeq;
  $("#post-busy").hidden = false;
  try {
    const desc = await postJSON(`/api/runs/${d.id}/postprocess`, state.post);
    if (seq !== postSeq || state.current?.id !== desc.id) return;
    state.current = desc;
    await viewer.updateSources({ cutout: desc.urls.cutout, mask: desc.urls.mask, overlay: desc.urls.overlay });
    updateDownloads();
    renderMetrics();
    state.history = state.history.map((r) => (r.id === desc.id ? desc : r));
    renderHistory();
  } catch (err) {
    toastError(err);
  } finally {
    if (seq === postSeq) $("#post-busy").hidden = true;
  }
}

// ───────────────────────── run button ─────────────────────────

/** True if the current "run" settings would give a different result from the one shown. */
function inferChanged() {
  const used = state.current?.info?.params;
  if (!used) return false;
  const p = state.infer;
  const keys = ["model", "resolution", "precision", "mask_interp"];
  if (p.resolution === "custom") keys.push("width", "height");
  if (p.resolution === "original" || modelById(p.model)?.dynamic) keys.push("max_side");
  if (p.precision === "fp32") keys.push("matmul_precision");
  if (effectiveDevice() !== state.current.info.device) return true;
  return keys.some((k) => String(used[k]) !== String(p[k]));
}

function updateRunButton() {
  const btn = $("#run-btn");
  const label = $("#run-label");
  const m = modelById(state.infer.model);
  const p = state.infer;
  const res = p.resolution === "auto" ? (m?.dynamic ? `≤${p.max_side}px` : fmtRes(m?.native)) : p.resolution === "custom" ? `${p.width}×${p.height}` : `${t("res.original").toLowerCase()} ≤${p.max_side}px`;
  const dev = effectiveDevice();
  const prec = dev === "cpu" ? "FP32" : (p.precision === "bf16" && !(dev === "cuda" && state.system?.bf16_supported) ? "fp16" : p.precision).toUpperCase();
  $("#run-sub").textContent = `${m?.label || p.model} · ${res} · ${deviceLabel(dev)} ${prec}`;
  btn.classList.remove("stale");
  if (state.view === "batch") {
    const n = state.batch.items.filter((i) => i.status === "queued").length;
    label.textContent = n ? tn("run.batch", n) : t("run.batchEmpty");
    btn.disabled = !n || !!state.batch.job || (m && !m.downloaded);
  } else {
    label.textContent = state.current ? t("run.rerun") : t("run.run");
    btn.disabled = state.busy || !state.current || (m && !m.downloaded);
    if (state.current && inferChanged()) {
      btn.classList.add("stale");
      label.textContent = t("run.stale");
    }
  }
  if (m && !m.downloaded) label.textContent = t("run.download");
}

function onRun() {
  if (state.view === "batch") return startBatch();
  if (state.current) runSingle({ source: `run:${state.current.id}` }, state.current.name);
}

// ───────────────────────── batch ─────────────────────────

function addBatchItems(items) {
  for (const it of items) state.batch.items.push({ key: ++state.batch.seq, status: "queued", ...it });
  if (!state.batch.job) renderBatch();
}

function renderBatch() {
  const items = state.batch.items;
  $("#batch-empty").hidden = items.length > 0;
  $("#batch-count").textContent = tn("batch.count", items.length);
  const grid = $("#batch-grid");
  grid.replaceChildren();
  for (const it of items) {
    const card = document.createElement("div");
    card.className = `batch-card ${it.status}`;
    card.innerHTML = `<div class="bimg checker"><img alt="" loading="lazy"></div><div class="binfo"><span class="bname"></span><span class="bstate"></span></div>`;
    card.querySelector("img").src = it.result ? it.result.urls.thumb : it.thumb;
    card.querySelector(".bname").textContent = it.name;
    card.querySelector(".bstate").textContent = t(`batch.${it.status}`);
    if (it.error) card.title = it.error;
    if (it.status === "queued" && !state.batch.job) {
      const x = document.createElement("button");
      x.className = "bremove";
      x.title = t("batch.remove");
      x.textContent = "×";
      x.addEventListener("click", () => {
        if (it.thumb?.startsWith("blob:")) URL.revokeObjectURL(it.thumb);
        state.batch.items = state.batch.items.filter((i) => i !== it);
        renderBatch();
      });
      card.querySelector(".binfo").append(x);
    }
    if (it.status === "done") card.addEventListener("click", () => openRun(it.result));
    grid.append(card);
  }
  updateRunButton();
}

async function startBatch() {
  const queued = state.batch.items.filter((i) => i.status === "queued");
  if (!queued.length || state.batch.job) return;
  // The server processes uploaded files first, then examples: same order here.
  const order = [...queued.filter((i) => i.file), ...queued.filter((i) => !i.file)];
  const fd = new FormData();
  for (const it of order) if (it.file) fd.append("files", it.file, it.file.name);
  fd.append("sources", JSON.stringify({ items: order.filter((i) => !i.file).map((i) => i.source) }));
  fd.append("params", JSON.stringify(state.infer));
  fd.append("post", JSON.stringify(state.post));
  $("#batch-zip").setAttribute("aria-disabled", "true");
  try {
    const job = await api("/api/batch", { method: "POST", body: fd });
    state.batch.job = job;
    for (const it of order) it.status = "running";
    renderBatch();
    pollBatch(job.id, order);
  } catch (err) {
    toastError(err);
  }
}

function renderBatchProgress(j, secs) {
  const bar = $("#batch-progress");
  bar.hidden = false;
  const pct = Math.round(j.progress * 100);
  $(".progress-bar", bar).style.width = `${pct}%`;
  $(".progress-label", bar).textContent = j.status === "done"
    ? t("batch.finished", { ok: j.results.length, errors: j.errors.length, secs: j.elapsed_s ?? secs })
    : t("batch.progress", { done: j.done, total: j.total, pct, secs });
}

function pollBatch(jobId, order) {
  const t0 = performance.now();
  const timer = setInterval(async () => {
    let j;
    try { j = await api(`/api/jobs/${jobId}`); } catch (err) { clearInterval(timer); state.batch.job = null; toastError(err); return; }
    for (const r of j.results) { const it = order[r.index]; if (it && it.status !== "done") { it.status = "done"; it.result = r; } }
    for (const e of j.errors) { const it = order[e.index]; if (it) { it.status = "error"; it.error = errorText(e); } }
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    state.batch.lastJob = { ...j, secs };
    renderBatchProgress(j, secs);
    if (j.status === "done" || j.status === "error") {
      clearInterval(timer);
      state.batch.job = null;
      if (j.zip) {
        const z = $("#batch-zip");
        z.href = j.zip;
        z.download = `excido_batch_${jobId}.zip`;
        z.removeAttribute("aria-disabled");
      }
      const ok = j.status === "done";
      toast(t(ok ? "batch.toast.done" : "batch.toast.failed"), {
        body: ok ? t("batch.finished", { ok: j.results.length, errors: j.errors.length, secs: j.elapsed_s ?? secs }) : j.message,
        kind: ok ? "ok" : "error",
      });
      refreshHistory().catch(() => {});
      refreshSystem();
    }
    renderBatch();
  }, 600);
}

// ───────────────────────── views, theme, language, input ─────────────────────────

function switchView(view) {
  state.view = view;
  for (const b of $$(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.view === view));
  $("#view-single").hidden = view !== "single";
  $("#view-batch").hidden = view !== "batch";
  if (view === "single" && state.current) viewer.fit();
  if (view === "batch") renderBatch();
  updateRunButton();
}

function themeLabel() {
  const mode = document.documentElement.dataset.theme || "system";
  $("#theme-toggle").title = t("theme.title", { mode: t(`theme.${mode}`) });
}

function cycleTheme() {
  const root = document.documentElement;
  const order = [undefined, "light", "dark"];
  const next = order[(order.indexOf(root.dataset.theme) + 1) % order.length];
  if (next) root.dataset.theme = next;
  else delete root.dataset.theme;
  try { next ? localStorage.setItem("excido.theme", next) : localStorage.removeItem("excido.theme"); } catch {}
  themeLabel();
}

/** Re-render everything that holds translated text. */
function refreshTexts() {
  applyI18n();
  for (const s of $$("#lang-toggle span")) s.classList.toggle("on", s.dataset.lang === getLang());
  for (const k of $$(".mod-key")) k.textContent = MOD;
  themeLabel();
  renderDeviceButtons();
  if (state.models.length) { renderModelSelect(); renderModelCard(); }
  syncResolutionUI();
  renderExamples();
  renderHistory();
  if (state.current) { renderMetrics(); updateDownloads(); }
  if (state.view === "batch") renderBatch();
  if (state.batch.lastJob && !$("#batch-progress").hidden) renderBatchProgress(state.batch.lastJob, state.batch.lastJob.secs);
  viewer.relabel();
  updateRunButton();
}

function imageFiles(list) {
  return [...(list || [])].filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(f.name));
}

function handleFiles(files) {
  const imgs = imageFiles(files);
  if (!imgs.length) return toast(t("toast.noImages"), { kind: "error" });
  if (state.view === "batch") {
    addBatchItems(imgs.map((f) => ({ name: f.name, file: f, thumb: URL.createObjectURL(f) })));
  } else {
    runSingle({ file: imgs[0] }, imgs[0].name);
    if (imgs.length > 1) toast(t("toast.onlyFirst", { n: imgs.length }), { body: t("toast.useBatch") });
  }
}

function setViewMode(mode) {
  for (const b of $$("#view-modes button")) b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  viewer.setMode(mode);
}

function bindGlobal() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => switchView(b.dataset.view)));
  $("#theme-toggle").addEventListener("click", cycleTheme);
  $("#lang-toggle").addEventListener("click", () => { setLang(getLang() === "en" ? "it" : "en"); refreshTexts(); });
  $("#run-btn").addEventListener("click", onRun);

  $("#view-modes").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (b) setViewMode(b.dataset.mode);
  });
  $("#zoom-in").addEventListener("click", () => viewer.zoomBy(1.25));
  $("#zoom-out").addEventListener("click", () => viewer.zoomBy(0.8));
  $("#zoom-value").addEventListener("click", () => (viewer.autoFit ? viewer.actualSize() : viewer.fit()));

  $("#file-input").addEventListener("change", (e) => { handleFiles(e.target.files); e.target.value = ""; });
  $("#batch-input").addEventListener("change", (e) => { handleFiles(e.target.files); e.target.value = ""; });
  $("#url-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const url = $("#url-input").value.trim();
    if (!url) return;
    if (state.view === "batch") switchView("single");
    runSingle({ url }, t("name.url"));
  });
  $("#batch-examples").addEventListener("click", () => {
    addBatchItems(state.examples.flatMap((g) => g.items).map((it) => ({ name: it.name, source: it.source, thumb: it.url })));
  });
  $("#batch-clear").addEventListener("click", () => {
    if (state.batch.job) return;
    for (const it of state.batch.items) if (it.thumb?.startsWith("blob:")) URL.revokeObjectURL(it.thumb);
    state.batch.items = [];
    state.batch.lastJob = null;
    $("#batch-progress").hidden = true;
    $("#batch-zip").setAttribute("aria-disabled", "true");
    renderBatch();
  });

  // Paste from the clipboard
  document.addEventListener("paste", (e) => {
    if (e.target.closest?.("input, textarea")) return;
    const files = imageFiles(e.clipboardData?.files);
    if (files.length) { e.preventDefault(); handleFiles(files); }
  });

  // Drag & drop anywhere in the window
  let depth = 0;
  const overlay = $("#drop-overlay");
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; depth++; overlay.hidden = false; });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) overlay.hidden = true; });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    overlay.hidden = true;
    handleFiles(e.dataTransfer.files);
  });

  // Keyboard
  const typing = (e) => e.target.closest?.("input, select, textarea");
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); onRun(); return; }
    if (typing(e) || e.ctrlKey || e.metaKey || e.altKey || state.view !== "single") return;
    const modes = { 1: "compare", 2: "cutout", 3: "mask", 4: "overlay", 5: "side" };
    if (modes[e.key]) setViewMode(modes[e.key]);
    else if (e.key === "f" || e.key === "F") viewer.fit();
    else if (e.key === "0") viewer.actualSize();
    else if (e.key === "+" || e.key === "=") viewer.zoomBy(1.25);
    else if (e.key === "-") viewer.zoomBy(0.8);
    else if (e.code === "Space" && !e.target.closest?.("button, summary, a")) {
      e.preventDefault();
      if (!e.repeat) { viewer.peek(true); $("#peek-badge").hidden = !state.current; }
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space") { viewer.peek(false); $("#peek-badge").hidden = true; }
  });

  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshSystem(); });
  setInterval(() => { if (!document.hidden && !state.busy) refreshSystem(); }, 8000);
}

// ───────────────────────── start ─────────────────────────

async function init() {
  loadSaved();
  bindControls();
  bindGlobal();
  refreshTexts();
  syncAllUI();
  try {
    await refreshSystem();
    await refreshModels();
    syncInferUI();
    await Promise.all([refreshExamples(), refreshHistory()]);
  } catch (err) {
    toastError(err);
  }
  updateRunButton();

  // Open in a working state: the latest result, or the first official example.
  if (state.history.length) {
    openRun(state.history[0], { restorePost: false });
  } else {
    const first = state.examples[0]?.items?.[0];
    if (first && modelById(state.infer.model)?.downloaded) runSingle({ source: first.source }, first.name);
  }
}

init();
