import os
import sys
import time
import onnx
from onnx import helper, numpy_helper
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod

print("Loading clean_backbone.onnx...")
m = onnx.load('/home/ubuntu/da3-npu/models/clean_backbone.onnx', load_external_data=False)

# Target: Extract PatchEmbed + Block 0..3 (4 blocks)
# Input: image [1, 3, 504, 504]
# Output: /model/backbone/blocks.3/Add_1_output_0 [1, 1297, 1024]

out_tensor = '/model/backbone/blocks.3/Add_1_output_0'
node_by_output = {o: n for n in m.graph.node for o in n.output}
inits_by_name = {i.name: i for i in m.graph.initializer}

chunk_nodes = []
chunk_inits = []
visited = set(['image'])
queue = [out_tensor]

while queue:
    curr = queue.pop(0)
    if curr in visited:
        continue
    visited.add(curr)
    if curr in inits_by_name:
        chunk_inits.append(inits_by_name[curr])
        continue
    if curr in node_by_output:
        n = node_by_output[curr]
        if n not in chunk_nodes:
            chunk_nodes.append(n)
            for inp in n.input:
                if inp and inp not in visited:
                    queue.append(inp)

chunk_node_set = set(id(n) for n in chunk_nodes)
chunk_nodes = [n for n in m.graph.node if id(n) in chunk_node_set]
chunk_inits = list({i.name: i for i in chunk_inits}.values())

out_vi = helper.make_tensor_value_info(out_tensor, onnx.TensorProto.FLOAT, [1, 1297, 1024])
chunk_graph = helper.make_graph(
    nodes=chunk_nodes,
    name='chunk_0_3',
    inputs=list(m.graph.input),
    outputs=[out_vi],
    initializer=chunk_inits,
)
chunk_model = helper.make_model(chunk_graph, opset_imports=m.opset_import)
chunk_fp32 = '/tmp/chunk_0_3_fp32.onnx'
chunk_qdq = '/tmp/chunk_0_3_qdq.onnx'
onnx.save(chunk_model, chunk_fp32)
print(f"Chunk model saved ({len(chunk_nodes)} nodes, {len(chunk_inits)} inits) to {chunk_fp32}!")

class DummyReader(CalibrationDataReader):
    def __init__(self):
        self.data = iter([{'image': np.random.randn(1, 3, 504, 504).astype(np.float32)}])
    def get_next(self):
        return next(self.data, None)

print("Quantizing chunk to QDQ...")
quantize_static(
    model_input=chunk_fp32,
    model_output=chunk_qdq,
    calibration_data_reader=DummyReader(),
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=['Conv', 'MatMul', 'LayerNormalization', 'Add'],
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax,
    per_channel=False,
)
print(f"Quantized chunk size: {os.path.getsize(chunk_qdq) / (1024*1024):.2f} MB")

# Compile to Context
chunk_ctx = '/tmp/chunk_0_3_ctx.onnx'
print(f"Compiling chunk to EPContext: {chunk_ctx}...")

qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/home/ubuntu/qairt/2.35.0.250530/lib/hexagon-v68/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

lib_name = 'QNNExecutionProvider'
ort.register_execution_provider_library(lib_name, qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == lib_name]

sess_options = ort.SessionOptions()
sess_options.add_session_config_entry('ep.context_enable', '1')
sess_options.add_session_config_entry('ep.context_file_path', chunk_ctx)
sess_options.add_session_config_entry('ep.context_embed_mode', '1')

ep_options = {
    'backend_path': qnn_ep.get_qnn_htp_path(),
    'soc_model': '498',
    'htp_arch': '68',
}
sess_options.add_provider_for_devices(devices, ep_options)

t0 = time.time()
sess = ort.InferenceSession(chunk_qdq, sess_options=sess_options)
t1 = time.time()
print(f"Chunk context compilation succeeded in {t1 - t0:.2f}s! Context size: {os.path.getsize(chunk_ctx) / (1024*1024):.2f} MB")

# Now run inference on NPU using the compiled context!
sess_infer = ort.InferenceSession(chunk_ctx, sess_options=sess_options)
t2 = time.time()
out = sess_infer.run(None, {'image': np.zeros((1, 3, 504, 504), dtype=np.float32)})
t3 = time.time()
print(f"Chunk NPU Inference SUCCESS! Latency: {(t3-t2)*1000:.1f}ms! Output shape: {out[0].shape}")
