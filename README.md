# Depth Anything 3 (DA3METRIC-LARGE) on Qualcomm Rubik Pi 3 NPU

## *IMPORTANT*: This is a proof of concept and is mostly made with LLM, use at own risk

This project provides an end-to-end, hardware-accelerated deployment of Depth Anything 3's flagship monocular metric depth model (**[DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)**, 335M parameters) on the Qualcomm Rubik Pi 3 (QCS6490 / RB3 Gen 2) utilizing the Hexagon v68 DSP NPU and ONNX Runtime. It includes both host-native execution and a self-contained Docker container.

---

## 1. Technical Architecture

### Hardware Constraints & Solutions
- **Hexagon v68 DSP Math**: The Hexagon v68 HTP processor has no native FP16 tensor core matrix units; it is a high-throughput INT8/UINT8 vector engine. Attempting FP16 QNN execution results in `QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE` (`3110`). The backbone is quantized using static UINT8 QDQ quantization calibrated on intermediate ViT activations.
- **FastRPC CMA Memory Ceiling (~124 MB)**: The Linux kernel reserves 172 MB of Contiguous Memory Allocation (CMA) pool (`/dev/fastrpc-cdsp`), of which ~124 MB is available for user allocations. The 24-block ViT-Large backbone (1.3 GB) cannot be loaded as a monolithic graph.
- **Fine 8-Chunk Subgraph Partitioning**: The 24 ViT-Large Transformer blocks are partitioned into 8 fine chunks (2 to 4 blocks each), each compiling to an embedded QNN EPContext model between 55 MB and 113 MB.
- **DMA Buffer Recycling via Subprocess**: Sequential chunks are executed in isolated subprocesses. When each subprocess exits, the kernel closes `/dev/fastrpc-cdsp` and instantly returns 100% of CMA pages to the kernel pool, avoiding virtual memory fragmentation and CMA exhaustion.
- **DPT Decoder Head**: The DPT RefineNet multi-scale decoder head operates on 4 multi-level feature taps (from Blocks 4, 11, 17, and 23) and executes on CPU in ~6 seconds, producing high-fidelity 1920x1080 metric depth output.

---

## 2. Quick Start: Docker Container

### Prerequisites
- Qualcomm Rubik Pi 3 running Ubuntu 24.04 with `/dev/fastrpc-cdsp` and `/dev/dma_heap`.
- Docker installed and active.

### Running Inference via Helper Script
```bash
./run_container.sh [path/to/image.png] [path/to/output_dir]
```
Example:
```bash
./run_container.sh data/sample.png output/
```

### Running Directly with `docker run`
```bash
docker run --rm \
  --device /dev/fastrpc-cdsp \
  --device /dev/dma_heap \
  -v /path/to/my_image.png:/input.png:ro \
  -v /path/to/my_output:/app/output \
  da3-metric-large-npu:latest /input.png -o /app/output
```

---

## 3. Host-Native Execution

Run directly using the configured virtual environment:
```bash
.venv/bin/python3 deploy/entrypoint.py data/sample.png -o output/ --models_dir deploy/models
```
Or run the root pipeline script:
```bash
.venv/bin/python3 run_pipeline.py data/sample.png
```

---

## 4. Output Artifacts

Each inference produces two files in the designated output directory:
1. `depth_metric.npy`: 32-bit floating point NumPy array resized to the original input resolution (e.g. 1920x1080), containing metric depth in meters.
2. `depth_vis.png`: 8-bit normalized grayscale visualization of the metric depth map.

### Benchmark Latency on Qualcomm Rubik Pi 3
- **NPU Pure Execution (All 8 ViT Chunks)**: ~5.6 seconds (~700 ms per chunk)
- **NPU Wall-Clock (including process spinup & context init)**: ~18.5 seconds
- **DPT Head (CPU)**: ~6.0 seconds
- **End-to-End Latency**: ~24.7 seconds
