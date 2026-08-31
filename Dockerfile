FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# libgl1/libglib2.0-0: required by opencv-python-headless at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
RUN uv pip install --system --no-cache .

# Run as a non-root, fixed-UID user so files written to the bind-mounted
# ./data (uploads, generated CSV/PNG) are owned by a regular user on the
# host instead of root.
RUN useradd --uid 1000 --create-home vbt
USER vbt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
