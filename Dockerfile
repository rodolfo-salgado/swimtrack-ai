FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM nvcr.io/nvidia/tensorrt:25.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY vendor/ByteTrack ./vendor/ByteTrack

# TensorRT and CUDA Python bindings are supplied by the NVIDIA base image.
# The venv intentionally sees those system packages while uv locks everything else.
RUN uv venv --system-site-packages /opt/venv && uv sync --locked --no-dev --no-cache

RUN useradd --create-home --uid 10001 swimtrack && mkdir -p /model-cache && chown swimtrack:swimtrack /model-cache
USER swimtrack

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/readyz', timeout=3)"]
CMD ["python", "-m", "uvicorn", "swimtrack_ai.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
