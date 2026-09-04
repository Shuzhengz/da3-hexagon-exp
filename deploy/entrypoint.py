import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
from PIL import Image
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 (DA3METRIC-LARGE) NPU Monocular Metric Depth Estimation on Qualcomm Rubik Pi 3"
    )
    parser.add_argument('input', nargs='?', default='/app/data/sample.png',
                        help='Path to input image file (default: /app/data/sample.png)')
    parser.add_argument('-o', '--output', default='/app/output',
                        help='Output directory to save depth map and array (default: /app/output)')
    parser.add_argument('--models_dir', default='/app/models',
                        help='Directory containing compiled QNN context models')
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = args.input
    output_dir = args.output
    models_dir = args.models_dir
    cache_dir = '/tmp/da3_cache'
    
    if not os.path.isfile(image_path):
        print(f"Error: Input image file '{image_path}' not found!", file=sys.stderr)
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    print("================================================================")
    print(" Depth Anything 3: DA3METRIC-LARGE Monocular Depth Estimation")
    print(" Target: Qualcomm Rubik Pi 3 (Hexagon v68 DSP NPU)")
    print("================================================================")
    print(f"Input image:      {image_path}")
    print(f"Output directory: {output_dir}")
    
    # 1. Preprocessing
    print("\n[1/4] Preprocessing input image...")
    raw_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    print(f"  Input image resolution: {orig_w}x{orig_h}")
    
    t_prep0 = time.time()
    img = raw_img.resize((504, 504), Image.Resampling.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    inp = np.expand_dims(arr, 0).astype(np.float32)
    
    np.save(os.path.join(cache_dir, 'chunk_act_0.npy'), inp)
    print(f"  Preprocessed in {(time.time()-t_prep0)*1000:.1f}ms")
    
    # 2. Sequential NPU Chunk Execution
    print("\n[2/4] Executing 8 ViT-Large backbone chunks on Qualcomm NPU...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chunk_script = os.path.join(script_dir, 'run_chunk.py')
    
    npu_start = time.time()
    chunk_latencies = []
    for i in range(8):
        cmd = [
            sys.executable,
            chunk_script,
            '--chunk', str(i),
            '--models_dir', models_dir,
            '--cache_dir', cache_dir
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FATAL: Error executing chunk {i} on NPU:\n{res.stderr}\n{res.stdout}", file=sys.stderr)
            sys.exit(res.returncode)
        dt = (time.time() - t0) * 1000
        chunk_latencies.append(dt)
        for line in res.stdout.splitlines():
            if 'completed in' in line:
                print(f"  {line.strip()}")
                
    total_npu_ms = (time.time() - npu_start) * 1000
    print(f"  All 8 chunks completed on Hexagon NPU in {sum(chunk_latencies):.1f}ms (wall-clock: {total_npu_ms:.1f}ms)")
    
    # 3. DPT Head Execution
    print("\n[3/4] Running DPT multi-scale depth decoder head on CPU...")
    t_head0 = time.time()
    head_path = os.path.join(models_dir, 'clean_head.onnx')
    head = ort.InferenceSession(head_path, providers=['CPUExecutionProvider'])
    
    feed = {
        '/model/backbone/blocks.4/Add_1_output_0': np.load(os.path.join(cache_dir, 'chunk_act_2.npy')),
        '/model/backbone/blocks.11/Add_1_output_0': np.load(os.path.join(cache_dir, 'chunk_act_4.npy')),
        '/model/backbone/blocks.17/Add_1_output_0': np.load(os.path.join(cache_dir, 'chunk_act_6.npy')),
        '/model/backbone/blocks.23/Add_1_output_0': np.load(os.path.join(cache_dir, 'chunk_act_8.npy')),
    }
    out = head.run(None, feed)
    head_ms = (time.time() - t_head0) * 1000
    print(f"  DPT Head completed in {head_ms:.1f}ms")
    
    # 4. Postprocessing & Output
    print("\n[4/4] Postprocessing and saving depth predictions...")
    depth = out[0].squeeze()
    
    # Bilinear resize to original image dimensions
    depth_img = Image.fromarray(depth).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    depth_resized = np.array(depth_img)
    
    # Save metric depth array
    npy_path = os.path.join(output_dir, 'depth_metric.npy')
    np.save(npy_path, depth_resized)
    
    # Save normalized visualization
    norm_depth = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-6)
    depth_uint8 = (norm_depth * 255.0).astype(np.uint8)
    vis_path = os.path.join(output_dir, 'depth_vis.png')
    Image.fromarray(depth_uint8).save(vis_path)
    
    # Cleanup cache
    shutil.rmtree(cache_dir, ignore_errors=True)
    
    total_wall_clock = (time.time() - t_prep0) * 1000
    print("\n================================================================")
    print(" Inference Summary & Statistics")
    print("================================================================")
    print(f"  Depth Map Output:     {vis_path}")
    print(f"  Metric Numpy Array:   {npy_path}")
    print(f"  Output Resolution:    {orig_w}x{orig_h}")
    print(f"  Min Depth (meters):   {depth_resized.min():.3f} m")
    print(f"  Max Depth (meters):   {depth_resized.max():.3f} m")
    print(f"  Mean Depth (meters):  {depth_resized.mean():.3f} m")
    print(f"  Unique Depth Values:  {len(np.unique(depth_resized))}")
    print(f"  Total Inference Time: {total_wall_clock:.1f}ms ({total_wall_clock/1000:.2f}s)")
    print("================================================================\n")

if __name__ == '__main__':
    main()
