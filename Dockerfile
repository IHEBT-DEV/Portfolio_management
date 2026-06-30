# ==============================================================================
# STAGE 1: Build & Dependency Compilation
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system compilation tools required for heavy C extensions (numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency definitions to leverage Docker caching layers
COPY requirements.txt .

# Install dependencies into a localized wheel storage directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# STAGE 2: Final Lean Runtime Setup
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Re-expose universal system logging output configurations instantly
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Copy installed site-packages directly from the builder stage layer
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy your clean lowercase/snake_case production directory files
COPY modules/ ./modules/
COPY services/ ./services/
COPY main.py .

# Create a non-root system user to securely isolate container runtime privileges
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE ${PORT}

# Fire up your production WSGI factory process server
CMD ["python", "main.py"]
