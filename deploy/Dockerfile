# syntax=docker/dockerfile:1
# Depth Anything 3 (DA3METRIC-LARGE) NPU Container for Qualcomm Rubik Pi 3 (Hexagon v68 DSP)
FROM ubuntu:24.04

LABEL maintainer="Antigravity" \
      description="Depth Anything 3 (DA3METRIC-LARGE) Monocular Metric Depth on Qualcomm Rubik Pi 3 NPU"

# Prevent interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libatomic1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Qualcomm FastRPC and DMA-buf heap userspace libraries
COPY libs/libcdsprpc.so* /usr/lib/aarch64-linux-gnu/
COPY libs/libdmabufheap.so* /usr/lib/aarch64-linux-gnu/
RUN ldconfig

# Set environment paths for Python and QNN DSP runtime
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.12/site-packages"

# Copy pre-configured virtual environment containing onnxruntime and onnxruntime-qnn
COPY venv /opt/venv

# Set working directory and copy application code and compiled QNN context models
WORKDIR /app
COPY models /app/models
COPY run_chunk.py /app/run_chunk.py
COPY entrypoint.py /app/entrypoint.py
COPY data/sample.png /app/data/sample.png

RUN mkdir -p /app/output

# Set entrypoint to run inference CLI
ENTRYPOINT ["/opt/venv/bin/python3", "/app/entrypoint.py"]
CMD ["/app/data/sample.png", "-o", "/app/output"]
