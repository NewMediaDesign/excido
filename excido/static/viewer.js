// Image viewer with zoom/pan and five modes:
// compare (before/after slider), cutout (checkerboard), mask, overlay, side (side by side).
// With a brush set, dragging paints a stroke (previewed on a canvas) instead of panning;
// Space+drag or the middle button still pan.
import { t } from "./i18n.js";

const MIN_SCALE = 0.02;
const MAX_SCALE = 16;

export class Viewer {
  constructor(stage, frame, { onZoom, onStroke } = {}) {
    this.stage = stage;
    this.frame = frame;
    this.onZoom = onZoom || (() => {});
    this.onStroke = onStroke || (() => {});
    this.brush = null; // { mode: "add" | "erase", size, hardness } while painting is on
    this.spacePan = false;
    this.preview = null; // canvas on top of the image, in image pixels
    this.ring = el("div", { class: "brush-ring", hidden: "" });
    stage.append(this.ring);
    this.data = null; // { original, cutout, mask, overlay, width, height }
    this.mode = "compare";
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.split = 0.5;
    this.peeking = false;
    this.autoFit = true;
    this.tags = null;
    this._bindPointer();
    new ResizeObserver(() => { if (this.autoFit && this.data) this.fit(); }).observe(stage);
  }

  // ───────────────────────── data and modes ─────────────────────────

  setData(data, { keepView = false } = {}) {
    const sameSize = this.data && this.data.width === data.width && this.data.height === data.height;
    this.data = data;
    this._render();
    if (!(keepView && sameSize)) this.fit();
    else this._apply();
  }

  /** Swap images after post-processing without flicker: preload first, then replace. */
  async updateSources(data) {
    const roles = ["cutout", "mask", "overlay"];
    await Promise.all(roles.map((r) => preload(data[r])));
    this.data = { ...this.data, ...data };
    for (const img of this.frame.querySelectorAll("img[data-role]")) {
      const url = this.data[img.dataset.role];
      if (url && img.getAttribute("src") !== url) img.src = url;
    }
  }

  setMode(mode) {
    this.mode = mode;
    if (this.data) {
      this._render();
      this.fit();
    }
  }

  peek(on) {
    if (!this.data || this.peeking === on) return;
    this.peeking = on;
    this._render();
    this._apply();
  }

  /** Refresh the translated Before/After labels (after a language switch). */
  relabel() {
    const [before, after] = this.tags?.children || [];
    if (before) before.textContent = t("stage.before");
    if (after) after.textContent = t("stage.after");
  }

  /** Turn painting on ({mode, size, hardness}) or off (null). */
  setBrush(brush) {
    this.brush = brush;
    this.stage.classList.toggle("painting", !!brush);
    if (!brush) this.ring.hidden = true;
    else this.ring.dataset.mode = brush.mode;
    this._sizeRing();
  }

  clearPreview() {
    this.preview?.getContext("2d").clearRect(0, 0, this.preview.width, this.preview.height);
  }

  clear() {
    this.data = null;
    this.frame.replaceChildren();
    this.tags?.remove();
    this.tags = null;
  }

  // ───────────────────────── zoom ─────────────────────────

  contentSize() {
    const { width: w, height: h } = this.data;
    return this.mode === "side" && !this.peeking ? { w: w * 2 + this._gap(), h } : { w, h };
  }

  fit() {
    if (!this.data) return;
    const r = this.stage.getBoundingClientRect();
    const { w, h } = this.contentSize();
    const pad = 32;
    const s = Math.min((r.width - pad * 2) / w, (r.height - pad * 2) / h, 4);
    this.scale = clamp(s, MIN_SCALE, MAX_SCALE);
    this.tx = (r.width - w * this.scale) / 2;
    this.ty = (r.height - h * this.scale) / 2;
    this.autoFit = true;
    this._apply();
  }

  actualSize() {
    const r = this.stage.getBoundingClientRect();
    this.zoomAt(1, r.width / 2, r.height / 2);
  }

  zoomBy(factor, cx, cy) {
    const r = this.stage.getBoundingClientRect();
    this.zoomAt(this.scale * factor, cx ?? r.width / 2, cy ?? r.height / 2);
  }

  zoomAt(next, cx, cy) {
    if (!this.data) return;
    next = clamp(next, MIN_SCALE, MAX_SCALE);
    this.tx = cx - (cx - this.tx) * (next / this.scale);
    this.ty = cy - (cy - this.ty) * (next / this.scale);
    this.scale = next;
    this.autoFit = false;
    this._apply();
  }

  // ───────────────────────── rendering ─────────────────────────

  _gap() {
    return Math.round(this.data.width * 0.04);
  }

  _layer(role, { checker = false } = {}) {
    const { width: w, height: h } = this.data;
    const layer = el("div", { class: "layer" + (checker ? " checker" : "") });
    layer.style.width = `${w}px`;
    layer.style.height = `${h}px`;
    layer.append(img(this.data[role], role, w, h));
    return layer;
  }

  _render() {
    const d = this.data;
    this.tags?.remove();
    this.tags = null;
    if (!d) return;
    let content;
    const mode = this.peeking ? "original" : this.mode;
    if (mode === "original") {
      content = this._layer("original");
    } else if (mode === "compare") {
      content = this._layer("original");
      const after = el("div", { class: "compare-after checker" });
      after.style.cssText = "position:absolute;inset:0";
      after.append(img(d.cutout, "cutout", d.width, d.height));
      const handle = el("div", { class: "split-handle" });
      handle.append(el("div", { class: "split-hit" }));
      content.append(after, handle);
      content.style.setProperty("--split", `${this.split * 100}%`);
      this.compareLayer = content;
      this.tags = el("div", { class: "compare-tags" });
      this.tags.innerHTML = '<span class="pane-label" style="left:12px"></span><span class="pane-label" style="left:auto;right:12px"></span>';
      this.stage.append(this.tags);
      this.relabel();
    } else if (mode === "cutout") {
      content = this._layer("cutout", { checker: true });
    } else if (mode === "mask") {
      content = this._layer("mask");
    } else if (mode === "overlay") {
      content = this._layer("overlay");
    } else {
      content = el("div", { class: "side-by-side" });
      content.style.gap = `${this._gap()}px`;
      content.append(this._layer("original"), this._layer("cutout", { checker: true }));
    }
    // Stroke preview canvas, on top of the image (single-image modes only).
    this.preview = null;
    if (content.classList.contains("layer")) {
      this.preview = el("canvas", { class: "brush-preview" });
      this.preview.width = d.width;
      this.preview.height = d.height;
      content.append(this.preview);
    }
    this.frame.replaceChildren(content);
  }

  _apply() {
    this.frame.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.scale})`;
    this.frame.style.setProperty("--inv-scale", String(1 / this.scale));
    for (const layer of this.frame.querySelectorAll(".layer")) {
      layer.classList.toggle("pixelated", this.scale >= 2);
    }
    this._sizeRing();
    this.onZoom(this.scale);
  }

  _sizeRing() {
    if (!this.brush) return;
    const px = Math.max(4, this.brush.size * this.scale);
    this.ring.style.width = this.ring.style.height = `${px}px`;
  }

  _toImage(e) {
    const r = this.stage.getBoundingClientRect();
    return [(e.clientX - r.left - this.tx) / this.scale, (e.clientY - r.top - this.ty) / this.scale];
  }

  _drawSegment(a, b) {
    const ctx = this.preview?.getContext("2d");
    if (!ctx) return;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.lineWidth = this.brush.size;
    ctx.strokeStyle = this.brush.mode === "erase" ? "rgba(255, 92, 64, 0.55)" : "rgba(141, 182, 242, 0.55)";
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0] + 0.01, b[1]);
    ctx.stroke();
  }

  // ───────────────────────── interaction ─────────────────────────

  _bindPointer() {
    const st = this.stage;
    let drag = null;

    st.addEventListener("wheel", (e) => {
      if (!this.data) return;
      e.preventDefault();
      const r = st.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
      this.zoomBy(factor, e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    st.addEventListener("pointerdown", (e) => {
      if (!this.data || (e.button !== 0 && e.button !== 1)) return;
      st.focus({ preventScroll: true });
      if (this.brush && this.preview && e.button === 0 && !this.spacePan) {
        e.preventDefault();
        const p = this._toImage(e);
        drag = { kind: "paint", points: [p], last: p };
        st.setPointerCapture(e.pointerId);
        this._drawSegment(p, p);
        return;
      }
      const onHandle = e.button === 0 && !this.brush && e.target.closest(".split-handle");
      drag = { kind: onHandle ? "split" : "pan", x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty };
      st.setPointerCapture(e.pointerId);
      if (drag.kind === "pan") st.classList.add("panning");
      else this._moveSplit(e.clientX);
    });
    st.addEventListener("pointermove", (e) => {
      if (this.brush) {
        const r = st.getBoundingClientRect();
        this.ring.hidden = false;
        this.ring.style.left = `${e.clientX - r.left}px`;
        this.ring.style.top = `${e.clientY - r.top}px`;
      }
      if (!drag) return;
      if (drag.kind === "paint") {
        const p = this._toImage(e);
        // Keep points about 1/8 of the brush apart: smooth strokes, small payloads.
        if (Math.hypot(p[0] - drag.last[0], p[1] - drag.last[1]) < Math.max(1, this.brush.size / 8)) return;
        this._drawSegment(drag.last, p);
        drag.points.push(p);
        drag.last = p;
        return;
      }
      if (drag.kind === "split") return this._moveSplit(e.clientX);
      this.tx = drag.tx + (e.clientX - drag.x);
      this.ty = drag.ty + (e.clientY - drag.y);
      this.autoFit = false;
      this._apply();
    });
    const end = () => {
      if (drag?.kind === "paint" && this.brush) {
        this.onStroke({ mode: this.brush.mode, size: this.brush.size, hardness: this.brush.hardness, points: drag.points });
      }
      drag = null;
      st.classList.remove("panning");
    };
    st.addEventListener("pointerleave", () => { this.ring.hidden = true; });
    st.addEventListener("pointerup", end);
    st.addEventListener("pointercancel", end);
    st.addEventListener("dblclick", () => { if (!this.brush) this.fit(); });
  }

  _moveSplit(clientX) {
    if (!this.compareLayer) return;
    const r = this.stage.getBoundingClientRect();
    const x = (clientX - r.left - this.tx) / this.scale / this.data.width;
    this.split = clamp(x, 0, 1);
    this.compareLayer.style.setProperty("--split", `${this.split * 100}%`);
  }
}

// ───────────────────────── helpers ─────────────────────────

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function el(tag, attrs = {}) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function img(src, role, w, h) {
  const i = el("img", { "data-role": role, alt: "", draggable: "false", decoding: "async" });
  i.width = w;
  i.height = h;
  i.style.width = `${w}px`;
  i.style.height = `${h}px`;
  i.src = src;
  return i;
}

function preload(url) {
  return new Promise((resolve) => {
    if (!url) return resolve();
    const i = new Image();
    i.onload = i.onerror = () => resolve();
    i.src = url;
  });
}
