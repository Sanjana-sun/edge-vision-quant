# Benchmark Results

FashionMNIST test set (10,000 images). CPU, qnnpack INT8 backend.

| Metric | FP32 | INT8 | Change |
|---|---|---|---|
| Test accuracy (%) | 91.55 | 91.42 | -0.13 pts |
| Model size (MB) | 0.385 | 0.102 | 3.78x smaller |
| Latency (ms/img) | 0.362 | 0.203 | 1.78x faster |

INT8 holds accuracy at **91.42%** (-0.13 pts vs FP32) while being **3.78x smaller** and **1.78x faster** per image.
