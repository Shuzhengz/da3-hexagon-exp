import os
import gc
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/home/ubuntu/qairt/2.35.0.250530/lib/hexagon-v68/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

lib_name = 'QNNExecutionProvider'
ort.register_execution_provider_library(lib_name, qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == lib_name]

sess_options = ort.SessionOptions()
ep_options = {
    'backend_path': qnn_ep.get_qnn_htp_path(),
    'soc_model': '498',
    'htp_arch': '68',
}
sess_options.add_provider_for_devices(devices, ep_options)

models_dir = '/home/ubuntu/da3-npu/models'

# Sample input
img = Image.open('/home/ubuntu/da3-npu/data/sample.png').convert('RGB')
img = img.resize((504, 504), Image.Resampling.BILINEAR)
arr = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
arr = np.transpose(arr, (2, 0, 1))
inp = np.expand_dims(arr, 0).astype(np.float32)

print("\n--- Running Sequential NPU Execution ---", flush=True)

# Chunk 0
print("Running Chunk 0 on NPU...", flush=True)
s0 = ort.InferenceSession(f'{models_dir}/chunk0_ctx.onnx', sess_options=sess_options)
t0 = time.time()
out0 = s0.run(None, {'image': inp})[0]
print(f"Chunk 0 done in {(time.time()-t0)*1000:.1f}ms! Out shape: {out0.shape}", flush=True)
del s0
gc.collect()

# Chunk 1
print("Running Chunk 1 on NPU...", flush=True)
s1 = ort.InferenceSession(f'{models_dir}/chunk1_ctx.onnx', sess_options=sess_options)
t0 = time.time()
out1 = s1.run(None, {'/model/backbone/blocks.4/Add_1_output_0': out0})[0]
print(f"Chunk 1 done in {(time.time()-t0)*1000:.1f}ms! Out shape: {out1.shape}", flush=True)
del s1
gc.collect()

# Chunk 2
print("Running Chunk 2 on NPU...", flush=True)
s2 = ort.InferenceSession(f'{models_dir}/chunk2_ctx.onnx', sess_options=sess_options)
t0 = time.time()
out2 = s2.run(None, {'/model/backbone/blocks.11/Add_1_output_0': out1})[0]
print(f"Chunk 2 done in {(time.time()-t0)*1000:.1f}ms! Out shape: {out2.shape}", flush=True)
del s2
gc.collect()

# Chunk 3
print("Running Chunk 3 on NPU...", flush=True)
s3 = ort.InferenceSession(f'{models_dir}/chunk3_ctx.onnx', sess_options=sess_options)
t0 = time.time()
out3 = s3.run(None, {'/model/backbone/blocks.17/Add_1_output_0': out2})[0]
print(f"Chunk 3 done in {(time.time()-t0)*1000:.1f}ms! Out shape: {out3.shape}", flush=True)
del s3
gc.collect()

# Head on CPU
print("Running DPT Head on CPU...", flush=True)
head = ort.InferenceSession(f'{models_dir}/clean_head.onnx', providers=['CPUExecutionProvider'])
feed = {
    '/model/backbone/blocks.4/Add_1_output_0': out0,
    '/model/backbone/blocks.11/Add_1_output_0': out1,
    '/model/backbone/blocks.17/Add_1_output_0': out2,
    '/model/backbone/blocks.23/Add_1_output_0': out3,
}
t0 = time.time()
out = head.run(None, feed)
print(f"Head done in {(time.time()-t0)*1000:.1f}ms!", flush=True)

depth = out[0].squeeze()
sky = out[1].squeeze()
print(f"\nALL 4 CHUNKS + HEAD EXECUTED SUCCESSFULLY!")
print(f"Depth min: {depth.min():.3f}, max: {depth.max():.3f}, mean: {depth.mean():.3f}")
print(f"Unique depth values: {len(np.unique(depth))}")
