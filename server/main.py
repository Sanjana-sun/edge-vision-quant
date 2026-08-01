"""Inference API for edge-vision-quant.

Serves the trained FP32 model and the true INT8-quantized model behind an HTTP
API. On each request it runs both models so the frontend can show, side by side,
that INT8 gives the same prediction while being smaller and faster. Also exposes
Grad-CAM saliency, a latency/throughput benchmark, and a confusion matrix.

Run from the repo root:
    pip install -r server/requirements.txt
    uvicorn server.main:app --port 8000
"""

import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import TinyConvNet, CLASS_NAMES  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
WEB = os.path.join(ROOT, "web")
MEAN, STD = 0.2860, 0.3530

app = FastAPI(title="edge-vision-quant inference API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATE = {"fp32": None, "int8": None, "samples": [], "fp32_mb": 0.0, "int8_mb": 0.0}


# ---------- model loading ----------
def _load_fp32(path):
    m = TinyConvNet()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def _load_int8(path):
    torch.backends.quantized.engine = "qnnpack"
    m = TinyConvNet()
    m.eval()
    m.fuse_modules()
    m.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


@app.on_event("startup")
def _startup():
    """Load models, but never crash the process: capture errors so the server
    still binds its port and reports the failure via /api/health."""
    import traceback
    try:
        fp32_path = os.path.join(ARTIFACTS, "model_fp32.pth")
        int8_path = os.path.join(ARTIFACTS, "model_int8.pth")
        if not (os.path.exists(fp32_path) and os.path.exists(int8_path)):
            raise RuntimeError("Missing model artifacts (artifacts/model_fp32.pth / model_int8.pth).")
        STATE["fp32"] = _load_fp32(fp32_path)
        STATE["int8"] = _load_int8(int8_path)
        STATE["fp32_mb"] = os.path.getsize(fp32_path) / (1024 * 1024)
        STATE["int8_mb"] = os.path.getsize(int8_path) / (1024 * 1024)
        manifest = os.path.join(WEB, "manifest.json")
        if os.path.exists(manifest):
            STATE["samples"] = json.load(open(manifest)).get("samples", [])
        STATE["error"] = None
        print("[startup] models loaded OK")
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print("[startup] FAILED:", STATE["error"])
        traceback.print_exc()


def _require_models():
    if STATE["fp32"] is None or STATE["int8"] is None:
        raise HTTPException(503, STATE.get("error") or "models not loaded")


# ---------- preprocessing / inference ----------
def _to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("L").resize((28, 28))
    px = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 255.0
    t = px.view(1, 1, 28, 28)
    return (t - MEAN) / STD


def _apply_noise(x, noise: float):
    if noise and noise > 0:
        g = torch.Generator().manual_seed(0)
        return x + noise * torch.randn(x.shape, generator=g)
    return x


def _infer(model, x, runs=25, warmup=3):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            logits = model(x)
        ms = 1000.0 * (time.perf_counter() - t0) / runs
    probs = F.softmax(logits[0], dim=0).tolist()
    top = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {"topIdx": top, "label": CLASS_NAMES[top], "probs": probs, "latencyMs": ms}


def _gradcam(x, class_idx):
    """Grad-CAM heatmap (28x28, 0..1) over the FP32 model's last conv block."""
    model = STATE["fp32"]
    acts = {}

    def hook(_m, _i, o):
        o.retain_grad()
        acts["a"] = o

    h = model.relu3.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(False)
    logits = model(xi)
    logits[0, class_idx].backward()
    h.remove()
    a = acts["a"]                                   # [1,64,7,7]
    w = a.grad.mean(dim=(2, 3), keepdim=True)       # [1,64,1,1]
    cam = F.relu((w * a).sum(dim=1, keepdim=True))  # [1,1,7,7]
    cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return [[round(v, 4) for v in row] for row in cam.detach().tolist()]


def _classify(x, true_idx=None):
    fp32 = _infer(STATE["fp32"], x)
    int8 = _infer(STATE["int8"], x)
    cam = _gradcam(x, fp32["topIdx"])
    return {
        "classNames": CLASS_NAMES,
        "fp32": fp32,
        "int8": int8,
        "agree": fp32["topIdx"] == int8["topIdx"],
        "gradcam": cam,
        "trueIdx": true_idx,
        "trueLabel": CLASS_NAMES[true_idx] if true_idx is not None else None,
        "sizeMB": {"fp32": round(STATE["fp32_mb"], 3), "int8": round(STATE["int8_mb"], 3)},
    }


# ---------- API ----------
@app.get("/api/health")
def health():
    loaded = STATE["fp32"] is not None and STATE["int8"] is not None
    return {
        "status": "ok" if loaded else "degraded",
        "fp32Loaded": STATE["fp32"] is not None,
        "int8Loaded": STATE["int8"] is not None,
        "numSamples": len(STATE["samples"]),
        "error": STATE.get("error"),
    }


@app.get("/api/samples")
def samples():
    return {"classNames": CLASS_NAMES, "samples": STATE["samples"]}


@app.get("/api/classify/{sample_idx}")
def classify_sample(sample_idx: int, noise: float = 0.0):
    _require_models()
    if not (0 <= sample_idx < len(STATE["samples"])):
        raise HTTPException(404, "sample not found")
    s = STATE["samples"][sample_idx]
    img = Image.open(os.path.join(WEB, s["file"]))
    return _classify(_apply_noise(_to_tensor(img), noise), s.get("trueIdx"))


@app.post("/api/classify")
async def classify_upload(file: UploadFile = File(...), noise: float = 0.0):
    _require_models()
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "could not read image")
    return _classify(_apply_noise(_to_tensor(img), noise))


@app.get("/api/benchmark")
def benchmark(runs: int = 200):
    _require_models()
    runs = max(20, min(runs, 1000))
    x = _to_tensor(Image.open(os.path.join(WEB, STATE["samples"][0]["file"]))) \
        if STATE["samples"] else torch.zeros(1, 1, 28, 28)

    def timed(model):
        lat = []
        with torch.no_grad():
            for _ in range(5):
                model(x)
            for _ in range(runs):
                t0 = time.perf_counter()
                model(x)
                lat.append((time.perf_counter() - t0) * 1000.0)
        mean = sum(lat) / len(lat)
        return {"meanMs": round(mean, 4), "imgPerSec": round(1000.0 / mean, 1),
                "latencies": [round(v, 4) for v in lat]}

    return {"runs": runs, "fp32": timed(STATE["fp32"]), "int8": timed(STATE["int8"])}


@app.get("/api/confusion")
def confusion():
    path = os.path.join(WEB, "confusion.json")
    if not os.path.exists(path):
        raise HTTPException(404, "confusion.json not generated; run export_web.py")
    return JSONResponse(json.load(open(path)))


# ---------- static frontend ----------
if os.path.isdir(WEB):
    app.mount("/samples", StaticFiles(directory=os.path.join(WEB, "samples")), name="samples")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB, "index.html"))

    @app.get("/app.js")
    def appjs():
        return FileResponse(os.path.join(WEB, "app.js"))


# ---------- entrypoint ----------
# Read the port in Python so no shell variable expansion is needed anywhere
# (Railway/Render/etc. inject PORT). Run with: python -m server.main
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT") or 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)
