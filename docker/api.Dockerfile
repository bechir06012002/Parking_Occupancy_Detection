# Shares its base image and dependency-install steps with worker.Dockerfile —
# build context is the repo root, only the final CMD differs between the
# two. Swaps in opencv-python-headless: pyproject.toml pins the
# GUI-enabled opencv-python build for scripts/annotate_spots.py's local
# interactive use, but containers never open a window.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer first, cached independently of application code changes.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY backend/app ./app
COPY backend/worker ./worker
COPY backend/alembic.ini ./
RUN uv sync --frozen --no-dev \
    && uv pip uninstall opencv-python \
    && uv pip install opencv-python-headless==5.0.0.93

EXPOSE 8000
# --no-sync: `uv run` re-syncs the venv against pyproject.toml/uv.lock by
# default, which would silently undo the opencv-headless swap above (and
# pull in the dev dependency group) on every container start.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
