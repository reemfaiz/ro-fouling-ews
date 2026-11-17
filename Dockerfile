# rofoul - RO fouling early-warning service
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# model artefacts, plant config and sample data (models/ and config/ can be
# replaced at run time with volume mounts -> swap models without rebuilding)
COPY models ./models
COPY config ./config
COPY data/sample ./data/sample

# Make sure the shipped model loads with the scikit-learn version installed in
# the image; if it does not (pickle incompatibility), rebuild it from the
# seeded synthetic dataset (~1 min) so the image is always self-consistent.
RUN python -c "from rofoul.models import GBMForecaster; GBMForecaster.load('models/v1')" \
    || (rofoul synth --out /tmp/synthetic && rofoul train /tmp/synthetic/plant_data.csv --out models/v1 --version v1 \
        && rm -rf /tmp/synthetic)

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# `docker run IMAGE`            -> REST API on :8000
# `docker run IMAGE predict ...` -> CLI
ENTRYPOINT ["rofoul"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
