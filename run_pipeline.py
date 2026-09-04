import os
import sys
import time
import subprocess
import numpy as np
from PIL import Image
import onnxruntime as ort

models_dir = '/home/ubuntu/da3-npu/models'
output_dir = '/home/ubuntu/da3-npu/output'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('/tmp/da3_cache', exist_ok=True)

image_path = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/da3-npu/data/sample.png'
print(f"Loading input image: {image_path}")
raw_img = Image.open(image_path).convert('RGB')
orig_w, orig_h = raw_img.size

# Preprocessing
img = raw_img.resize((504, 504), Image.Resampling.BILINEAR)
arr = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
arr = np.transpose(arr, (2, 0, 1))
inp = np.expand_dims(arr, 0).astype(np.float32)
np.save('/tmp/da3_cache/chunk_act_0.npy', inp)

print("\n=== Executing Qualcomm Rubik Pi 3 NPU Pipeline ===")
total_start = time.time()

# Run each NPU chunk in an isolated subprocess to ensure 100% DMA buffer recycling
chunk_latencies = []
for i in range(8):
    t0 = time.time()
    cmd = [
        sys.executable,
        '/home/ubuntu/da3-npu/run_chunk_step.py',
        str(i)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in chunk {i}:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    chunk_latencies.append(dt)
    # Parse latency from stdout
    for line in res.stdout.splitlines():
        if 'completed in' in line:
            print(f"  {line.strip()}")

print(f"\nAll 8 NPU chunks completed! Total NPU wall-clock time: {sum(chunk_latencies):.1f} ms")

# Run Head on CPU
print("Running DPT Head on CPU...")
t0 = time.time()
head = ort.InferenceSession(f'{models_dir}/clean_head.onnx', providers=['CPUExecutionProvider'])
out0 = np.load('/tmp/da3_cache/chunk_act_2.npy')  # Block 4
out1 = np.load('/tmp/da3_cache/chunk_act_4.npy')  # Block 11
out2 = np.load('/tmp/da3_cache/chunk_act_6.npy')  # Block 17
out3 = np.load('/tmp/da3_cache/chunk_act_8.npy')  # Block 23

feed = {
    '/model/backbone/blocks.4/Add_1_output_0': out0,
    '/model/backbone/blocks.11/Add_1_output_0': out1,
    '/model/backbone/blocks.17/Add_1_output_0': out2,
    '/model/backbone/blocks.23/Add_1_output_0': out3,
}
out = head.run(None, feed)
t1 = time.time()
head_ms = (t1 - t0) * 1000
print(f"DPT Head completed in {head_ms:.1f} ms!")

total_ms = (time.time() - total_start) * 1000
print(f"\n==========================================")
print(f"Total End-to-End Pipeline Latency: {total_ms:.1f} ms ({total_ms/1000:.2f}s)")
print(f"==========================================")

depth = out[0].squeeze()
sky = out[1].squeeze()

depth_img = Image.fromarray(depth).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
depth_resized = np.array(depth_img)

print(f"\nDepth Output Statistics:")
print(f"  Resolution:    {orig_w}x{orig_h}")
print(f"  Min depth:     {depth_resized.min():.3f} meters")
print(f"  Max depth:     {depth_resized.max():.3f} meters")
print(f"  Mean depth:    {depth_resized.mean():.3f} meters")
print(f"  Unique values: {len(np.unique(depth_resized))}")

# Normalize and colorize
norm_depth = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-6)
depth_uint8 = (norm_depth * 255.0).astype(np.uint8)

vis_path = f'{output_dir}/depth_vis.png'
Image.fromarray(depth_uint8).save(vis_path)
print(f"Saved depth visualization to: {vis_path}")

npy_path = f'{output_dir}/depth_metric.npy'
np.save(npy_path, depth_resized)
print(f"Saved metric depth array to:   {npy_path}")
