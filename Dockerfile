FROM python:3.11-slim

WORKDIR /app

# PyTorch needs libgomp1 (OpenMP), which the slim image omits.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# CPU-only torch + API deps
RUN pip install --no-cache-dir -r server/requirements.txt

ENV OMP_NUM_THREADS=1

EXPOSE 8000
# artifacts/model_*.pth and web/ are committed, so this runs out of the box.
# Shell form so $PORT (injected by Render/Railway/HF/Fly) is honored; defaults to 8000.
CMD uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}
