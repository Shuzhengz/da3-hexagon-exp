import os
import sys
import time
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/home/ubuntu/qairt/2.35.0.250530/lib/hexagon-v68/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

lib_name = 'QNNExecutionProvider'
ort.register_execution_provider_library(lib_name, qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == lib_name]

model_in = '/home/ubuntu/da3-npu/models/clean_backbone_all_qdq.onnx'
model_ctx = '/home/ubuntu/da3-npu/models/backbone_ctx.onnx'

print(f'Compiling {model_in} to QNN Context model: {model_ctx}...')
sess_options = ort.SessionOptions()
sess_options.add_session_config_entry('ep.context_enable', '1')
sess_options.add_session_config_entry('ep.context_file_path', model_ctx)
sess_options.add_session_config_entry('ep.context_embed_mode', '1')

ep_options = {
    'backend_path': qnn_ep.get_qnn_htp_path(),
    'soc_model': '498',
    'htp_arch': '68',
}
sess_options.add_provider_for_devices(devices, ep_options)

t0 = time.time()
sess = ort.InferenceSession(model_in, sess_options=sess_options)
t1 = time.time()
print(f'Context compilation completed in {t1 - t0:.2f}s!')
if os.path.exists(model_ctx):
    print(f'Output context model size: {os.path.getsize(model_ctx) / (1024*1024):.2f} MB')
