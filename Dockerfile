FROM python:3.12-slim

# libgl1/libglib2.0-0: required by opencv-python-headless at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

# Run as a non-root, fixed-UID user so files written to the bind-mounted
# ./data (uploads, generated CSV/PNG) are owned by a regular user on the
# host instead of root.
RUN useradd --uid 1000 --create-home vbt
USER vbt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
