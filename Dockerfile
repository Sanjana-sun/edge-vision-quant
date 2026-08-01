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
# The app reads PORT from the environment itself (server/main.py __main__),
# so no shell variable expansion is required anywhere.
CMD ["python", "-m", "server.main"]
