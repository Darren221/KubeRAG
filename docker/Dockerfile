# syntax=docker/dockerfile:1.7

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1

# uv is the dependency manager; install it first so layer cache survives
# pyproject.toml changes.
COPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /usr/local/bin/

# Build in /app so the venv's shebangs (e.g. /app/.venv/bin/python) point at
# the same location they'll resolve to in the runtime image. Otherwise console
# scripts like `uvicorn` would carry shebangs referencing the builder's WORKDIR
# and fail with "no such file or directory" at runtime.
WORKDIR /app

# Copy only what affects dependency resolution. This layer cache hits on
# every code-only change, which is most of them.
COPY pyproject.toml uv.lock README.md ./

# Install runtime deps only (no [dev]).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the project itself.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---- Stage 2: runtime ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user. Production containers should never run as root.
RUN groupadd -r app \
    && useradd -r -g app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /data \
    && chown -R app:app /app /data

WORKDIR /app

# Copy only the populated venv + source from builder. No uv, no pip cache,
# no build toolchain ships in the runtime image.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

# Seed script — used by the compose `seed` service to populate empty indexes
# on first boot. Adds <1 KB to the image.
COPY --chown=app:app scripts/ /app/scripts/

USER app

# Default storage paths point at the mounted volume.
ENV KUBERAG_CHROMA_PATH=/data/chroma \
    KUBERAG_BM25_PATH=/data/bm25.pkl \
    KUBERAG_RAW_DOCS_PATH=/data/raw

EXPOSE 8000

# Factory form so create_app() runs at server-boot time and lifespan triggers.
CMD ["uvicorn", "kuberag.api:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
