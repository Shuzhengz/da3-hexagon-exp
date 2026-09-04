import onnx
from onnx import helper, numpy_helper
import numpy as np

print("Loading clean_backbone.onnx...")
m = onnx.load('/home/ubuntu/da3-npu/models/clean_backbone.onnx', load_external_data=False)

node_by_output = {o: n for n in m.graph.node for o in n.output}
inits_by_name = {i.name: i for i in m.graph.initializer}

chunks_spec = [
    {
        'name': 'chunk0',
        'inputs': [('image', onnx.TensorProto.FLOAT, [1, 3, 504, 504])],
        'outputs': [('/model/backbone/blocks.4/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'stop_inputs': set(['image']),
    },
    {
        'name': 'chunk1',
        'inputs': [('/model/backbone/blocks.4/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'outputs': [('/model/backbone/blocks.11/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'stop_inputs': set(['/model/backbone/blocks.4/Add_1_output_0']),
    },
    {
        'name': 'chunk2',
        'inputs': [('/model/backbone/blocks.11/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'outputs': [('/model/backbone/blocks.17/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'stop_inputs': set(['/model/backbone/blocks.11/Add_1_output_0']),
    },
    {
        'name': 'chunk3',
        'inputs': [('/model/backbone/blocks.17/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'outputs': [('/model/backbone/blocks.23/Add_1_output_0', onnx.TensorProto.FLOAT, [1, 1297, 1024])],
        'stop_inputs': set(['/model/backbone/blocks.17/Add_1_output_0']),
    },
]

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
    out_path = f'/home/ubuntu/da3-npu/models/{c_name}_fp32.onnx'
    onnx.save(model, out_path)
    print(f"Saved {c_name} to {out_path}: {len(ordered_nodes)} nodes, {len(ordered_inits)} inits")

print("All chunks extracted successfully!")
