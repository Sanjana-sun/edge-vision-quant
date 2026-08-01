/* On-Device INT8 Image Classifier — browser demo (ONNX Runtime Web) */
(() => {
  const MEAN = 0.2860, STD = 0.3530;
  let CLASS_NAMES = [], SAMPLES = [], session = null, activeThumb = null;

  const $ = (id) => document.getElementById(id);
  const statusEl = $("status");

  // single-threaded wasm from CDN (no cross-origin-isolation needed)
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

  async function init() {
    try {
      const manifest = await (await fetch("manifest.json")).json();
      CLASS_NAMES = manifest.classNames;
      SAMPLES = manifest.samples;
      buildGallery();
      session = await ort.InferenceSession.create("model_fp32.onnx", {
        executionProviders: ["wasm"],
      });
      statusEl.textContent = "Ready — pick an image.";
      statusEl.style.color = "var(--accent2)";
      // classify the first sample automatically
      if (SAMPLES.length) selectSample(0, document.querySelector(".thumb"));
    } catch (e) {
      statusEl.textContent = "Failed to load model: " + e.message;
      console.error(e);
    }
  }

  function buildGallery() {
    const g = $("gallery");
    g.innerHTML = "";
    SAMPLES.forEach((s, i) => {
      const b = document.createElement("button");
      b.className = "thumb";
      b.title = s.trueLabel;
      const img = document.createElement("img");
      img.src = s.file;
      b.appendChild(img);
      b.addEventListener("click", () => selectSample(i, b));
      g.appendChild(b);
    });
  }

  function setActive(el) {
    if (activeThumb) activeThumb.classList.remove("active");
    activeThumb = el;
    if (el) el.classList.add("active");
  }

  // draw a source (img or canvas) into the 28x28 view + return normalized tensor data
  function preprocessFromImage(imgEl) {
    const view = $("view");
    const vctx = view.getContext("2d");
    vctx.clearRect(0, 0, 28, 28);
    vctx.drawImage(imgEl, 0, 0, 28, 28);
    const { data } = vctx.getImageData(0, 0, 28, 28);
    const input = new Float32Array(28 * 28);
    for (let p = 0; p < 28 * 28; p++) {
      const gray = data[p * 4] / 255; // r channel (grayscale png)
      input[p] = (gray - MEAN) / STD;
    }
    return input;
  }

  async function classify(input, truthIdx) {
    if (!session) return;
    const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
    const t0 = performance.now();
    const out = await session.run({ input: tensor });
    const t1 = performance.now();
    const logits = Array.from(out.logits.data);
    const probs = softmax(logits);
    render(probs, t1 - t0, truthIdx);
  }

  function softmax(a) {
    const m = Math.max(...a);
    const ex = a.map((v) => Math.exp(v - m));
    const s = ex.reduce((x, y) => x + y, 0);
    return ex.map((v) => v / s);
  }

  function render(probs, ms, truthIdx) {
    const idx = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const topIdx = idx[0][1];
    $("predLabel").textContent = CLASS_NAMES[topIdx];
    $("predMeta").textContent = `${(probs[topIdx] * 100).toFixed(1)}% confidence · ${ms.toFixed(1)} ms in-browser`;

    const truth = $("truth");
    if (truthIdx != null) {
      const ok = truthIdx === topIdx;
      truth.innerHTML = ok
        ? `<span class="ok">✓ correct</span> (actual: ${CLASS_NAMES[truthIdx]})`
        : `<span class="no">✗ actual: ${CLASS_NAMES[truthIdx]}</span>`;
    } else {
      truth.textContent = "";
    }

    const bars = $("bars");
    bars.innerHTML = "";
    idx.slice(0, 5).forEach(([p, i], rank) => {
      const row = document.createElement("div");
      row.className = "bar-row" + (rank === 0 ? " top" : "");
      row.innerHTML =
        `<span class="name">${CLASS_NAMES[i]}</span>` +
        `<span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
        `<span class="pct">${(p * 100).toFixed(0)}%</span>`;
      bars.appendChild(row);
    });
  }

  function selectSample(i, el) {
    setActive(el);
    const s = SAMPLES[i];
    const img = new Image();
    img.onload = () => classify(preprocessFromImage(img), s.trueIdx);
    img.src = s.file;
  }

  // upload handler
  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setActive(null);
    const img = new Image();
    img.onload = () => classify(preprocessFromImage(img), null);
    img.src = URL.createObjectURL(f);
  });

  init();
})();
