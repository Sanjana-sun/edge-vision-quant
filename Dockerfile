FROM python:3.11-slim

WORKDIR /app
COPY . /app

# CPU-only torch + API deps
RUN pip install --no-cache-dir -r server/requirements.txt

EXPOSE 8000
# artifacts/model_*.pth and web/ are committed, so this runs out of the box
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
