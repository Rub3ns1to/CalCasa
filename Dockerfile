# Imagen multi-stage optimizada para FastAPI + XGBoost + Pillow + fpdf2
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .


FROM python:3.11-slim

LABEL org.opencontainers.image.title="CalCasa · Tasador Inmobiliario Multimodal (Madrid)"
LABEL org.opencontainers.image.authors="Rubén García Núñez"

# libgomp1 es necesario para OpenMP de XGBoost; fonts-dejavu-core para Unicode (€) en informes PDF
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root UID 1000 (estándar seguro y compatible con Hugging Face Spaces)
RUN useradd --create-home --uid 1000 --shell /bin/sh appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser models ./models

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=3)"

CMD ["sh", "-c", "uvicorn tasador.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
