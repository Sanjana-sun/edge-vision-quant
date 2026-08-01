/* edge-vision-quant — full-stack demo frontend */
(() => {
  const API = ""; // same origin (served by FastAPI); set to http://localhost:8000 if opened standalone
  const $ = (id) => document.getElementById(id);
  const statusEl = $("status");
  let CLASS_NAMES = [], SAMPLES = [], current = null, lastCam = null;

  async function jget(path) {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(path + " -> " + r.status);
    return r.json();
  }

  async function init() {
    try {
      await jget("/api/health");
      const s = await jget("/api/samples");
      CLASS_NAMES = s.classNames; SAMPLES = s.samples;
      buildGallery();
      statusEl.textContent = "Connected — both models loaded.";
      statusEl.className = "status ok";
      loadConfusion();
      if (SAMPLES.length) selectSample(0, document.querySelector(".thumb"));
    } catch (e) {
      statusEl.textContent = "Cannot reach API. Start it with: uvicorn server.main:app --port 8000";
      statusEl.className = "status err";
      console.error(e);
    }
    wireControls();
    setupDraw();
  }

  // ---------- gallery ----------
  function buildGallery() {
    const g = $("gallery"); g.innerHTML = "";
    SAMPLES.forEach((s, i) => {
      const b = document.createElement("button");
      b.className = "thumb"; b.title = s.trueLabel;
      const img = document.createElement("img"); img.src = s.file; b.appendChild(img);
      b.addEventListener("click", () => selectSample(i, b));
      g.appendChild(b);
    });
  }
  let activeThumb = null;
  function setActive(el){ if(activeThumb) activeThumb.classList.remove("active"); activeThumb=el; if(el) el.classList.add("active"); }

  // ---------- classify ----------
  function noise() { return parseFloat($("noise").value) || 0; }

  async function selectSample(i, el) {
    setActive(el);
    current = { type: "sample", idx: i };
    drawView(SAMPLES[i].file);
    await run();
  }

  async function classifyBlob(blob) {
    setActive(null);
    current = { type: "blob", blob };
    drawView(URL.createObjectURL(blob));
    await run();
  }

  async function run() {
    if (!current) return;
    $("verdictText").textContent = "Classifying…";
    try {
      let data;
      if (current.type === "sample") {
        data = await jget(`/api/classify/${current.idx}?noise=${noise()}`);
      } else {
        const fd = new FormData();
        fd.append("file", current.blob, "input.png");
        const r = await fetch(`${API}/api/classify?noise=${noise()}`, { method: "POST", body: fd });
        data = await r.json();
      }
      render(data);
    } catch (e) { $("verdictText").textContent = "Error: " + e.message; }
  }

  function render(d) {
    lastCam = d.gradcam;
    renderModel("fp32", d.fp32, d);
    renderModel("int8", d.int8, d);
    const badge = $("agreeBadge");
    badge.className = "badge " + (d.agree ? "ok" : "no");
    badge.textContent = d.agree ? "✓ predictions agree" : "✗ predictions differ";
    $("verdictText").textContent = d.agree
      ? `Both predict “${d.fp32.label}”.`
      : `FP32: ${d.fp32.label} · INT8: ${d.int8.label}`;
    const t = $("truth");
    if (d.trueIdx != null) {
      const ok = d.trueIdx === d.int8.topIdx;
      t.innerHTML = ok ? `<span class="ok">✓ actual class: ${d.trueLabel}</span>`
                       : `<span class="no">actual class: ${d.trueLabel}</span>`;
    } else t.textContent = "";
    drawHeat();
  }

  function renderModel(key, m, d) {
    $(key + "Pred").textContent = m.label;
    const sz = d.sizeMB[key];
    $(key + "Chips").innerHTML =
      `<span class="chip"><b>${(m.probs[m.topIdx]*100).toFixed(1)}%</b> conf</span>` +
      `<span class="chip"><b>${m.latencyMs.toFixed(2)}</b> ms</span>` +
      `<span class="chip"><b>${sz.toFixed(3)}</b> MB</span>`;
    const idx = m.probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]).slice(0,5);
    $(key + "Bars").innerHTML = idx.map(([p,i],r)=>
      `<div class="bar-row${r===0?" top":""}"><span class="name">${CLASS_NAMES[i]}</span>`+
      `<span class="bar-track"><span class="bar-fill" style="width:${(p*100).toFixed(1)}%"></span></span>`+
      `<span class="pct">${(p*100).toFixed(0)}%</span></div>`).join("");
  }

  // ---------- image + Grad-CAM ----------
  function drawView(src) {
    const img = new Image();
    img.onload = () => { const c=$("view").getContext("2d"); c.clearRect(0,0,28,28); c.drawImage(img,0,0,28,28); };
    img.src = src;
  }
  function drawHeat() {
    const heat = $("heat"); heat.style.display = $("camToggle").checked && lastCam ? "block" : "none";
    if (!lastCam || !$("camToggle").checked) return;
    const ctx = heat.getContext("2d"); const im = ctx.createImageData(28,28);
    for (let y=0;y<28;y++) for (let x=0;x<28;x++){
      const v = lastCam[y][x]; const p=(y*28+x)*4;
      im.data[p]=255; im.data[p+1]=Math.round(200*(1-v)); im.data[p+2]=40; im.data[p+3]=Math.round(230*v);
    }
    ctx.putImageData(im,0,0);
  }

  // ---------- controls ----------
  function wireControls() {
    $("noise").addEventListener("input", () => { $("noiseVal").textContent = noise().toFixed(2); });
    $("noise").addEventListener("change", run);
    $("camToggle").addEventListener("change", drawHeat);
    $("file").addEventListener("change", (e)=>{ const f=e.target.files[0]; if(f) classifyBlob(f); });
    $("runBench").addEventListener("click", runBench);
    $("classifyDraw").addEventListener("click", classifyDrawing);
    $("clearDraw").addEventListener("click", clearDraw);
  }

  // ---------- draw canvas ----------
  let drawing=false, dctx=null;
  function setupDraw(){
    const c=$("draw"); dctx=c.getContext("2d"); clearDraw();
    const pos=(e)=>{const r=c.getBoundingClientRect();const t=e.touches?e.touches[0]:e;return [t.clientX-r.left,t.clientY-r.top];};
    const start=(e)=>{drawing=true;const[x,y]=pos(e);dctx.beginPath();dctx.moveTo(x,y);e.preventDefault();};
    const move=(e)=>{if(!drawing)return;const[x,y]=pos(e);dctx.lineTo(x,y);dctx.stroke();e.preventDefault();};
    const end=()=>{drawing=false;};
    dctx.strokeStyle="#fff";dctx.lineWidth=10;dctx.lineCap="round";dctx.lineJoin="round";
    c.addEventListener("mousedown",start);c.addEventListener("mousemove",move);window.addEventListener("mouseup",end);
    c.addEventListener("touchstart",start);c.addEventListener("touchmove",move);c.addEventListener("touchend",end);
  }
  function clearDraw(){ dctx.fillStyle="#000"; dctx.fillRect(0,0,140,140); }
  function classifyDrawing(){
    const t=document.createElement("canvas"); t.width=28;t.height=28;
    t.getContext("2d").drawImage($("draw"),0,0,28,28);
    t.toBlob((b)=>classifyBlob(b),"image/png");
  }

  // ---------- benchmark ----------
  async function runBench() {
    $("benchStatus").textContent = "running…";
    try {
      const d = await jget("/api/benchmark?runs=200");
      $("benchStats").style.display = "grid"; $("histWrap").style.display = "block";
      const sp = (d.fp32.meanMs / d.int8.meanMs);
      $("bSpeed").textContent = sp.toFixed(2) + "×";
      $("bFp32").textContent = `${d.fp32.meanMs.toFixed(3)} · ${d.fp32.imgPerSec}`;
      $("bInt8").textContent = `${d.int8.meanMs.toFixed(3)} · ${d.int8.imgPerSec}`;
      const max = Math.max(...d.fp32.latencies, ...d.int8.latencies);
      hist($("histFp"), d.fp32.latencies, max, "");
      hist($("histInt"), d.int8.latencies, max, "int");
      $("benchStatus").textContent = `${d.runs} runs each`;
    } catch (e) { $("benchStatus").textContent = "error: " + e.message; }
  }
  function hist(el, arr, max, cls) {
    el.innerHTML = arr.map(v=>`<span class="b ${cls}" style="height:${Math.max(3,(v/max)*100)}%"></span>`).join("");
  }

  // ---------- confusion matrix ----------
  async function loadConfusion() {
    try {
      const d = await jget("/api/confusion");
      $("cmOverall").textContent = `Overall INT8 accuracy: ${d.overallAcc}%.`;
      renderCM(d); renderPerClass(d);
    } catch (e) { $("cmOverall").textContent = "(confusion matrix unavailable)"; }
  }
  function renderCM(d) {
    const n = d.classNames.length, m = d.matrix;
    const grid = $("cmGrid");
    grid.style.gridTemplateColumns = `64px repeat(${n},1fr)`;
    const short = d.classNames.map(c => c.replace("/top","").slice(0,4));
    let html = `<div class="hd"></div>` + short.map(s=>`<div class="hd">${s}</div>`).join("");
    for (let i=0;i<n;i++){
      const rowSum = m[i].reduce((a,b)=>a+b,0);
      html += `<div class="hd" style="justify-content:flex-end;padding-right:4px">${short[i]}</div>`;
      for (let j=0;j<n;j++){
        const v=m[i][j], frac=v/rowSum;
        const col = i===j ? `rgba(86,214,176,${0.15+0.85*frac})` : `rgba(240,120,138,${Math.min(1,frac*3)})`;
        const txt = v>0 ? v : "";
        html += `<div class="cell" style="background:${col}" title="${d.classNames[i]} → ${d.classNames[j]}: ${v}">${txt}</div>`;
      }
    }
    grid.innerHTML = html;
  }
  function renderPerClass(d) {
    const el=$("perClass");
    const rows = d.classNames.map((c,i)=>({c,a:d.perClassAcc[i]})).sort((a,b)=>a.a-b.a);
    el.innerHTML = `<div class="hint" style="margin-bottom:6px">Per-class accuracy (weakest first)</div>` +
      rows.map(r=>`<div class="pc-row"><span class="name">${r.c}</span>`+
        `<span class="pc-track"><span class="pc-fill" style="width:${r.a}%"></span></span>`+
        `<span class="pct">${r.a}%</span></div>`).join("");
  }

  init();
})();
