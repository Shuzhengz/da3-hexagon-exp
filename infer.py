import os
import sys
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# 1. Setup paths
qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/home/ubuntu/qairt/2.35.0.250530/lib/hexagon-v68/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

lib_name = 'QNNExecutionProvider'
ort.register_execution_provider_library(lib_name, qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == lib_name]

# EP options for precompiled context loading
sess_options = ort.SessionOptions()
ep_options = {
    'backend_path': qnn_ep.get_qnn_htp_path(),
    'soc_model': '498',
    'htp_arch': '68',
}
sess_options.add_provider_for_devices(devices, ep_options)

models_dir = '/home/ubuntu/da3-npu/models'
print("Initializing Qualcomm Hexagon NPU pipeline (Depth Anything 3 Metric Large)...")
t_load_0 = time.time()

c0 = ort.InferenceSession(f'{models_dir}/chunk0_ctx.onnx', sess_options=sess_options)
c1 = ort.InferenceSession(f'{models_dir}/chunk1_ctx.onnx', sess_options=sess_options)
c2 = ort.InferenceSession(f'{models_dir}/chunk2_ctx.onnx', sess_options=sess_options)
c3 = ort.InferenceSession(f'{models_dir}/chunk3_ctx.onnx', sess_options=sess_options)
head = ort.InferenceSession(f'{models_dir}/clean_head.onnx', providers=['CPUExecutionProvider'])

t_load_1 = time.time()
print(f"All models loaded in {t_load_1 - t_load_0:.2f}s!")

# Preprocessing
image_path = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/da3-npu/data/sample.png'
print(f"Reading input image: {image_path}")
raw_img = Image.open(image_path).convert('RGB')
orig_w, orig_h = raw_img.size

img = raw_img.resize((504, 504), Image.Resampling.BILINEAR)
arr = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
arr = np.transpose(arr, (2, 0, 1))
inp = np.expand_dims(arr, 0).astype(np.float32)

print("\n--- Running Inference ---")
t_start = time.time()

# Chunk 0 on NPU (PatchEmbed + Blocks 0..4)
t0 = time.time()
out0 = c0.run(None, {'image': inp})[0]
t1 = time.time()
print(f"Chunk 0 (NPU - PatchEmbed + Blocks 0..4):  {(t1-t0)*1000:6.1f} ms")

# Chunk 1 on NPU (Blocks 5..11)
t0 = time.time()
out1 = c1.run(None, {'/model/backbone/blocks.4/Add_1_output_0': out0})[0]
t1 = time.time()
print(f"Chunk 1 (NPU - Blocks 5..11):              {(t1-t0)*1000:6.1f} ms")

# Chunk 2 on NPU (Blocks 12..17)
t0 = time.time()
out2 = c2.run(None, {'/model/backbone/blocks.11/Add_1_output_0': out1})[0]
t1 = time.time()
print(f"Chunk 2 (NPU - Blocks 12..17):             {(t1-t0)*1000:6.1f} ms")

# Chunk 3 on NPU (Blocks 18..23)
t0 = time.time()
out3 = c3.run(None, {'/model/backbone/blocks.17/Add_1_output_0': out2})[0]
t1 = time.time()
print(f"Chunk 3 (NPU - Blocks 18..23):             {(t1-t0)*1000:6.1f} ms")

# Head on CPU
t0 = time.time()
feed = {
    '/model/backbone/blocks.4/Add_1_output_0': out0,
    '/model/backbone/blocks.11/Add_1_output_0': out1,
    '/model/backbone/blocks.17/Add_1_output_0': out2,
    '/model/backbone/blocks.23/Add_1_output_0': out3,
}
out = head.run(None, feed)
t1 = time.time()
print(f"Head    (CPU - DPT RefineNet + Regression):{(t1-t0)*1000:6.1f} ms")

t_end = time.time()
total_ms = (t_end - t_start) * 1000
print(f"\nTotal Pipeline Inference Time: {total_ms:.1f} ms ({total_ms/1000:.2f}s)")

depth = out[0].squeeze() # [504, 504]
sky = out[1].squeeze()   # [504, 504]

# Metric scaling: in DA3 metric models, the model predicts depth normalized by focal length / 300
# Resize depth back to original image size
depth_img = Image.fromarray(depth).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
depth_resized = np.array(depth_img)

print(f"\nOutput Statistics:")
print(f"  Depth map shape: {depth_resized.shape}")
print(f"  Min depth:       {depth_resized.min():.3f} meters")
print(f"  Max depth:       {depth_resized.max():.3f} meters")
print(f"  Mean depth:      {depth_resized.mean():.3f} meters")
print(f"  Unique values:   {len(np.unique(depth_resized))}")

# Colormap depth map visualization
norm_depth = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-6)
depth_uint8 = (norm_depth * 255.0).astype(np.uint8)

# Save visualization using simple plasma colormap or grayscale
os.makedirs('/home/ubuntu/da3-npu/output', exist_ok=True)
out_vis_path = '/home/ubuntu/da3-npu/output/depth_vis.png'
Image.fromarray(depth_uint8).save(out_vis_path)
print(f"Saved depth visualization to {out_vis_path}")

np.save('/home/ubuntu/da3-npu/output/depth_metric.npy', depth_resized)
print("Saved metric depth array to /home/ubuntu/da3-npu/output/depth_metric.npy")
