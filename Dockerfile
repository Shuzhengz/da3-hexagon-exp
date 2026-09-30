# syntax=docker/dockerfile:1
# Depth Anything 3 (DA3METRIC-LARGE) NPU Container for Qualcomm Rubik Pi 3 (Hexagon v68 DSP)
# Repository: da3-hexagon-exp
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="da3-hexagon-exp" \
      org.opencontainers.image.description="Depth Anything 3 (DA3METRIC-LARGE) on Qualcomm Rubik Pi 3 Hexagon v68 DSP NPU" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libatomic1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Qualcomm FastRPC and DMA-buf heap userspace libraries
COPY libs/libcdsprpc.so* /usr/lib/aarch64-linux-gnu/
COPY libs/libdmabufheap.so* /usr/lib/aarch64-linux-gnu/
RUN ldconfig

# Set up Python virtual environment and install dependencies
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Set environment paths for Python and QNN DSP runtime
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.12/site-packages"

# Copy compiled QNN context models and runtime scripts
COPY deploy/models /app/models
COPY entrypoint.py /app/entrypoint.py
COPY run_chunk.py /app/run_chunk.py
COPY data/sample.png /app/data/sample.png

RUN mkdir -p /app/output

# Set entrypoint to run inference CLI
ENTRYPOINT ["/opt/venv/bin/python3", "/app/entrypoint.py"]
CMD ["/app/data/sample.png", "-o", "/app/output"]
