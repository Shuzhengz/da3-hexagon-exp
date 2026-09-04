import os
import sys
import time
import numpy as np
from PIL import Image
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod

models_dir = '/home/ubuntu/da3-npu/models'
os.makedirs(models_dir, exist_ok=True)

print("Loading clean_backbone.onnx...")
m = onnx.load(f'{models_dir}/clean_backbone.onnx', load_external_data=False)

node_by_output = {o: n for n in m.graph.node for o in n.output}
inits_by_name = {i.name: i for i in m.graph.initializer}

cut_blocks = [2, 4, 7, 11, 14, 17, 20, 23]
chunks_spec = []

prev_tensor = 'image'
prev_elem = onnx.TensorProto.FLOAT
prev_shape = [1, 3, 504, 504]

for i, cb in enumerate(cut_blocks):
    out_tensor = f'/model/backbone/blocks.{cb}/Add_1_output_0'
    out_elem = onnx.TensorProto.FLOAT
    out_shape = [1, 1297, 1024]
    
    spec = {
        'name': f'chunk_{i}',
        'inputs': [(prev_tensor, prev_elem, prev_shape)],
        'outputs': [(out_tensor, out_elem, out_shape)],
        'stop_inputs': set([prev_tensor]),
    }
    chunks_spec.append(spec)
    
    prev_tensor = out_tensor
    prev_shape = out_shape

print("Extracting 8 subgraphs...")
for spec in chunks_spec:
    c_name = spec['name']
    queue = [o[0] for o in spec['outputs']]
    stop = spec['stop_inputs']
    visited = set(stop)
    nodes = []
    inits = []
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        if curr in inits_by_name:
            inits.append(inits_by_name[curr])
            continue
        if curr in node_by_output:
            n = node_by_output[curr]
            if n not in nodes:
                nodes.append(n)
                for inp in n.input:
                    if inp and inp not in visited:
                        queue.append(inp)
    node_set = set(id(n) for n in nodes)
    ordered_nodes = [n for n in m.graph.node if id(n) in node_set]
    ordered_inits = list({i.name: i for i in inits}.values())
    
    in_vis = [helper.make_tensor_value_info(name, elem, shape) for name, elem, shape in spec['inputs']]
    out_vis = [helper.make_tensor_value_info(name, elem, shape) for name, elem, shape in spec['outputs']]
    
    g = helper.make_graph(ordered_nodes, c_name, in_vis, out_vis, ordered_inits)
    model = helper.make_model(g, opset_imports=m.opset_import)
    out_path = f'{models_dir}/{c_name}_fp32.onnx'
    onnx.save(model, out_path)
    print(f"Saved {c_name} to {out_path}: {len(ordered_nodes)} nodes, {len(ordered_inits)} inits")

print("\nExtracting intermediate activations for calibration...")
img = Image.open('/home/ubuntu/da3-npu/data/sample.png').convert('RGB')
img = img.resize((504, 504), Image.Resampling.BILINEAR)
arr = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std
arr = np.transpose(arr, (2, 0, 1))
cur_act = np.expand_dims(arr, 0).astype(np.float32)

calib_inputs = []
for i, spec in enumerate(chunks_spec):
    in_name = spec['inputs'][0][0]
    calib_inputs.append({in_name: cur_act})
    sess = ort.InferenceSession(f'{models_dir}/chunk_{i}_fp32.onnx', providers=['CPUExecutionProvider'])
    cur_act = sess.run(None, {in_name: cur_act})[0]

class SingleDictReader(CalibrationDataReader):
    def __init__(self, d):
        self.data = iter([d])
    def get_next(self):
        return next(self.data, None)

qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/home/ubuntu/qairt/2.35.0.250530/lib/hexagon-v68/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

lib_name = 'QNNExecutionProvider'
ort.register_execution_provider_library(lib_name, qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == lib_name]

print("\nQuantizing and compiling all 8 chunks...")
for i, spec in enumerate(chunks_spec):
    c_name = spec['name']
    fp32_path = f'{models_dir}/{c_name}_fp32.onnx'
    qdq_path = f'{models_dir}/{c_name}_qdq.onnx'
    ctx_path = f'{models_dir}/{c_name}_ctx.onnx'
    
    print(f"\n--- {c_name} ---")
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
    print(f"Quantized in {time.time() - t0:.1f}s, size: {os.path.getsize(qdq_path)/(1024*1024):.2f} MB")
    
    if os.path.exists(ctx_path):
        os.remove(ctx_path)
        
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
    ctx_size = os.path.getsize(ctx_path) / (1024*1024)
    print(f"Context compiled in {time.time() - t0:.1f}s, size: {ctx_size:.2f} MB")

print("\nALL 8 CHUNKS BUILT AND COMPILED!")
