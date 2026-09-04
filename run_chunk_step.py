import os
import sys
import time
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

chunk_idx = int(sys.argv[1])
models_dir = '/home/ubuntu/da3-npu/models'
cache_dir = '/tmp/da3_cache'
os.makedirs(cache_dir, exist_ok=True)

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

ctx_path = f'{models_dir}/chunk_{chunk_idx}_ctx.onnx'
sess = ort.InferenceSession(ctx_path, sess_options=sess_options)

in_name = sess.get_inputs()[0].name
inp = np.load(f'{cache_dir}/chunk_act_{chunk_idx}.npy')

t0 = time.time()
out = sess.run(None, {in_name: inp})[0]
t1 = time.time()
print(f"Chunk {chunk_idx} NPU execution completed in {(t1-t0)*1000:.1f}ms! Shape: {out.shape}", flush=True)

np.save(f'{cache_dir}/chunk_act_{chunk_idx+1}.npy', out)

