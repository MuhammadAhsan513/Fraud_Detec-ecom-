FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=model.joblib \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    REPORTS_DIR=/app/reports

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY features.py detect_anomalies.py model.joblib ./
COPY app ./app

RUN useradd -m appuser \
    && mkdir -p /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Single worker: the report store keeps state in this process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
