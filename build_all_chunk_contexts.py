import os
import sys
import time
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod

qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/home/ubuntu/qairt/2.35.0.250530/lib/hexagon-v68/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

lib_name = 'QNNExecutionProvider'
ort.register_execution_provider_library(lib_name, qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == lib_name]

img = Image.open('/home/ubuntu/da3-npu/data/sample.png').convert('RGB')
img = img.resize((504, 504), Image.Resampling.BILINEAR)
arr = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
arr = np.transpose(arr, (2, 0, 1))
sample_inp = np.expand_dims(arr, 0).astype(np.float32)

# Step 1: Run FP32 chunks sequentially to gather exact calibration activations
print("Gathering intermediate activations for calibration...")
c0_fp32 = ort.InferenceSession('/home/ubuntu/da3-npu/models/chunk0_fp32.onnx', providers=['CPUExecutionProvider'])
c1_fp32 = ort.InferenceSession('/home/ubuntu/da3-npu/models/chunk1_fp32.onnx', providers=['CPUExecutionProvider'])
c2_fp32 = ort.InferenceSession('/home/ubuntu/da3-npu/models/chunk2_fp32.onnx', providers=['CPUExecutionProvider'])

act0 = c0_fp32.run(None, {'image': sample_inp})[0]
act1 = c1_fp32.run(None, {'/model/backbone/blocks.4/Add_1_output_0': act0})[0]
act2 = c2_fp32.run(None, {'/model/backbone/blocks.11/Add_1_output_0': act1})[0]

calib_inputs = [
    {'image': sample_inp},
    {'/model/backbone/blocks.4/Add_1_output_0': act0},
    {'/model/backbone/blocks.11/Add_1_output_0': act1},
    {'/model/backbone/blocks.17/Add_1_output_0': act2},
]

class SingleDictReader(CalibrationDataReader):
    def __init__(self, d):
        self.data = iter([d])
    def get_next(self):
        return next(self.data, None)

for i in range(4):
    fp32_path = f'/home/ubuntu/da3-npu/models/chunk{i}_fp32.onnx'
    qdq_path = f'/home/ubuntu/da3-npu/models/chunk{i}_qdq.onnx'
    ctx_path = f'/home/ubuntu/da3-npu/models/chunk{i}_ctx.onnx'
    
    print(f"\n================ Processing Chunk {i} ================")
    # 1. Quantize
    if not os.path.exists(qdq_path):
        print(f"Quantizing {fp32_path} -> {qdq_path}...")
        t0 = time.time()
        quantize_static(
            model_input=fp32_path,
            model_output=qdq_path,
            calibration_data_reader=SingleDictReader(calib_inputs[i]),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=['Conv', 'MatMul', 'LayerNormalization', 'Add'],
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=False,
        )
        print(f"Quantization finished in {time.time() - t0:.1f}s! Size: {os.path.getsize(qdq_path)/(1024*1024):.2f} MB")
    else:
        print(f"Using existing {qdq_path}")

    # 2. Compile to Context
    if os.path.exists(ctx_path):
        os.remove(ctx_path)
        
    print(f"Compiling {qdq_path} -> {ctx_path}...")
    sess_options = ort.SessionOptions()
    sess_options.add_session_config_entry('ep.context_enable', '1')
    sess_options.add_session_config_entry('ep.context_file_path', ctx_path)
    sess_options.add_session_config_entry('ep.context_embed_mode', '1')
    
    ep_options = {
        'backend_path': qnn_ep.get_qnn_htp_path(),
        'soc_model': '498',
        'htp_arch': '68',
    }
    sess_options.add_provider_for_devices(devices, ep_options)
    
    t0 = time.time()
    sess = ort.InferenceSession(qdq_path, sess_options=sess_options)
    print(f"Compilation finished in {time.time() - t0:.1f}s! Context size: {os.path.getsize(ctx_path)/(1024*1024):.2f} MB")

print("\nALL 4 CHUNKS COMPILED SUCCESSFULLY!")
