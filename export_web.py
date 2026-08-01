import json, torch, numpy as np
from model import TinyConvNet, CLASS_NAMES
from torchvision import datasets

# --- load FP32 model ---
m = TinyConvNet()
state = torch.load("artifacts/model_fp32.pth", map_location="cpu")
m.load_state_dict(state)
m.eval()

# --- export to ONNX (opset 13, batch 1, 1x28x28) ---
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    m, dummy, "web/model_fp32.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13, dynamic_axes=None,
)
print("ONNX exported ->", "web/model_fp32.onnx")

# --- pull one raw test image per class (10) + a couple extra ---
test = datasets.FashionMNIST(root="./data", train=False, download=True)
data = test.data.numpy()        # uint8 [N,28,28], raw pixels
targets = test.targets.numpy()

manifest = []
try:
    from PIL import Image
    have_pil = True
except Exception:
    have_pil = False

picked = {}
order = []
for i in range(len(targets)):
    c = int(targets[i])
    if c not in picked:
        picked[c] = i
        order.append(c)
    if len(picked) == 10:
        break

for idx, c in enumerate(sorted(picked)):
    i = picked[c]
    img = data[i]  # 28x28 uint8
    fname = f"samples/s{idx}.png"
    if have_pil:
        Image.fromarray(img, mode="L").save(f"web/{fname}")
    manifest.append({"file": fname, "trueLabel": CLASS_NAMES[c], "trueIdx": c})

with open("web/manifest.json", "w") as f:
    json.dump({"classNames": CLASS_NAMES, "mean": 0.2860, "std": 0.3530, "samples": manifest}, f, indent=2)
print("samples:", len(manifest), "| PIL:", have_pil)
