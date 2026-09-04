import os
import sys
import time
import argparse
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

parser = argparse.ArgumentParser()
parser.add_argument('--chunk', type=int, required=True)
parser.add_argument('--models_dir', type=str, default='/app/models')
parser.add_argument('--cache_dir', type=str, default='/tmp/da3_cache')
args = parser.parse_args()

chunk_idx = args.chunk
models_dir = args.models_dir
cache_dir = args.cache_dir

qnn_dir = os.path.dirname(qnn_ep.get_library_path())
os.environ['ADSP_LIBRARY_PATH'] = f'{qnn_dir};/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp'

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

ctx_path = os.path.join(models_dir, f'chunk_{chunk_idx}_ctx.onnx')
sess = ort.InferenceSession(ctx_path, sess_options=sess_options)

in_name = sess.get_inputs()[0].name
inp = np.load(os.path.join(cache_dir, f'chunk_act_{chunk_idx}.npy'))

t0 = time.time()
out = sess.run(None, {in_name: inp})[0]
t1 = time.time()
print(f"Chunk {chunk_idx} NPU execution completed in {(t1-t0)*1000:.1f}ms! Shape: {out.shape}", flush=True)

np.save(os.path.join(cache_dir, f'chunk_act_{chunk_idx+1}.npy'), out)
