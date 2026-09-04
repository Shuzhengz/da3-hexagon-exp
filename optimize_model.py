import os
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

def optimize_da3_model(input_path, output_path):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    # 1. Set batch dim to 1
    inp = graph.input[0]
    inp.type.tensor_type.shape.dim[0].dim_value = 1
    inp.type.tensor_type.shape.dim[0].ClearField('dim_param')

    # 2. Add static constant initializers
    inits_to_add = {
        'da3_static_patch_embed_shape': np.array([1, 1024, 1296], dtype=np.int64),
        'da3_static_pos_embed_scales': np.array([1.0, 1.0, 0.975609756097561, 0.975609756097561], dtype=np.float32),
        'da3_static_qkv_shape': np.array([1, 1297, 3, 16, 64], dtype=np.int64),
        'da3_static_attn_out_shape': np.array([1, 1297, 1024], dtype=np.int64),
        'da3_static_attn_scale': np.array(0.3535533905932738, dtype=np.float16),
        'da3_static_head_slice_shape': np.array([1, 1296, 1024], dtype=np.int64),
        'da3_static_head_spatial_shape': np.array([1, 1024, 36, 36], dtype=np.int64),
        'da3_static_head_final_shape': np.array([1, 1, 504, 504], dtype=np.int64),
        'da3_static_refinenet4_sizes': np.array([1, 256, 36, 36], dtype=np.int64),
        'da3_static_refinenet3_sizes': np.array([1, 256, 72, 72], dtype=np.int64),
        'da3_static_refinenet2_sizes': np.array([1, 256, 144, 144], dtype=np.int64),
        'da3_static_refinenet1_sizes': np.array([1, 256, 288, 288], dtype=np.int64),
        'da3_static_head_resize_sizes': np.array([1, 128, 504, 504], dtype=np.int64),
    }

    for name, arr in inits_to_add.items():
        graph.initializer.append(numpy_helper.from_array(arr, name=name))

    # 3. Patch nodes
    for n in graph.node:
        if n.name == '/model/backbone/patch_embed/Reshape':
            n.input[1] = 'da3_static_patch_embed_shape'
        elif n.name == '/model/backbone/Concat_3':
            n.input[0] = 'model.model.backbone.pretrained.cls_token'
        elif n.name == '/model/backbone/Resize':
            n.input[2] = 'da3_static_pos_embed_scales'
        elif '/attn/Reshape' in n.name and n.op_type == 'Reshape':
            if n.name.endswith('/attn/Reshape'):
                n.input[1] = 'da3_static_qkv_shape'
            elif n.name.endswith('/attn/Reshape_1'):
                n.input[1] = 'da3_static_attn_out_shape'
        elif '/attn/Mul' in n.name and n.op_type == 'Mul':
            if n.name.endswith('/attn/Mul'):
                # Mul(Q, scale)
                n.input[1] = 'da3_static_attn_scale'
            elif n.name.endswith('/attn/Mul_1'):
                # Mul(K, scale)
                n.input[1] = 'da3_static_attn_scale'
        elif n.name in ['/model/head/Reshape', '/model/head/Reshape_1', '/model/head/Reshape_2', '/model/head/Reshape_3']:
            n.input[1] = 'da3_static_head_slice_shape'
        elif n.name in ['/model/head/Reshape_4', '/model/head/Reshape_5', '/model/head/Reshape_6', '/model/head/Reshape_7']:
            n.input[1] = 'da3_static_head_spatial_shape'
        elif n.name in ['/model/head/Reshape_8', '/model/head/Reshape_9']:
            n.input[1] = 'da3_static_head_final_shape'
        elif n.name == '/model/head/refinenet4/Resize':
            n.input[3] = 'da3_static_refinenet4_sizes'
        elif n.name == '/model/head/refinenet3/Resize':
            n.input[3] = 'da3_static_refinenet3_sizes'
        elif n.name == '/model/head/refinenet2/Resize':
            n.input[3] = 'da3_static_refinenet2_sizes'
        elif n.name == '/model/head/refinenet1/Resize':
            n.input[3] = 'da3_static_refinenet1_sizes'
        elif n.name == '/model/head/Resize':
            n.input[3] = 'da3_static_head_resize_sizes'

    # 4. Prune dead nodes backwards from outputs
    print("Performing dead code elimination...")
    required_tensors = set(o.name for o in graph.output)
    producer_map = {}
    for n in graph.node:
        for out in n.output:
            producer_map[out] = n

    reachable_nodes = set()
    queue = list(required_tensors)
    visited_tensors = set(queue)

    while queue:
        t = queue.pop()
        if t in producer_map:
            node = producer_map[t]
            reachable_nodes.add(node.name)
            for inp_t in node.input:
                if inp_t and inp_t not in visited_tensors:
                    visited_tensors.add(inp_t)
                    queue.append(inp_t)

    print(f"Reachable nodes: {len(reachable_nodes)} out of {len(graph.node)}")

    # Filter graph.node keeping original topological order
    new_nodes = [n for n in graph.node if n.name in reachable_nodes]
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Filter initializers
    used_initializers = set()
    for n in new_nodes:
        for inp_t in n.input:
            used_initializers.add(inp_t)
    new_inits = [i for i in graph.initializer if i.name in used_initializers]
    print(f"Retained {len(new_inits)} of {len(graph.initializer)} initializers.")
    del graph.initializer[:]
    graph.initializer.extend(new_inits)

    # Clear old value_info and run shape inference
    del graph.value_info[:]
    print("Running static shape inference...")
    inferred_model = onnx.shape_inference.infer_shapes(model, data_prop=True)

    print(f"Saving optimized static model to {output_path}...")
    onnx.save(inferred_model, output_path)
    print(f"Successfully saved {output_path}! File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    in_file = '/home/ubuntu/da3-npu/models/model_fp16.onnx'
    out_file = '/home/ubuntu/da3-npu/models/model_fp16_static.onnx'
    optimize_da3_model(in_file, out_file)
