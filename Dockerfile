# ── BUILD STAGE ──────────────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a local directory
COPY requirements.txt .

# Optimize: Use CPU-only versions of torch if GPU is not strictly required.
# This reduces image size by ~2GB.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
    -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ── RUNTIME STAGE ────────────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/bin:$PATH"

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

# Install ONLY necessary runtime system libraries
# (PyMuPDF and Torch may need shared libs, but slim-bookworm is usually sufficient)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY src/ ./src/
# Ensure data directory exists but is empty (to be mounted)
RUN mkdir -p data/pdfs

# Create and switch to non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Default command (API)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
