FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Map python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Prevent JAX from eating 100% of GPU memory
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Force JAX to use CUDA
ENV JAX_PLATFORMS=cuda,cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py"]
