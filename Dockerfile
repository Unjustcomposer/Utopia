# Stage 1: Build the frontend
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/ ./
RUN npm ci
RUN npm run build

# Stage 2: Backend and final image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and curl
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user 'utopia'
RUN useradd -m -u 1000 utopia

WORKDIR /app

# Set environment variables
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    JAX_PLATFORMS=cuda,cpu \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Copy frontend dist from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Set ownership to utopia user
RUN chown -R utopia:utopia /app

# Switch to non-root user
USER utopia

# Expose port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
