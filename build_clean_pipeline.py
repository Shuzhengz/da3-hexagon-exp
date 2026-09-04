import onnx
from onnx import helper, numpy_helper
import numpy as np

print("Loading model_fp32_nomatconv.onnx...")
m = onnx.load('/home/ubuntu/da3-npu/models/model_fp32_nomatconv.onnx', load_external_data=False)

node_by_name = {n.name: n for n in m.graph.node}

# Static initializers
init1 = numpy_helper.from_array(np.array([1, 37, 37, 1024], dtype=np.int64), name='da3_static_pos_grid_shape')
init2 = numpy_helper.from_array(np.array([1, 1296, 1024], dtype=np.int64), name='da3_static_patch_pos_shape')
init_feat_norm_shape = numpy_helper.from_array(np.array([1, 1, 1297, 1024], dtype=np.int64), name='da3_static_feat_norm_shape')
m.graph.initializer.extend([init1, init2, init_feat_norm_shape])

# 1. Static patch embed
node_by_name['/model/backbone/patch_embed/proj/Conv'].input[0] = 'image'
node_by_name['/model/backbone/Reshape_3'].input[1] = 'da3_static_pos_grid_shape'
node_by_name['/model/backbone/Reshape_4'].input[1] = 'da3_static_patch_pos_shape'

# 2. Block 0
node_by_name['/model/backbone/blocks.0/norm1/LayerNormalization'].input[0] = '/model/backbone/Add_output_0'
node_by_name['/model/backbone/blocks.0/Add'].input[0] = '/model/backbone/Add_output_0'

# 3. Blocks 1 to 23
for b in range(1, 24):
    prev_out = f'/model/backbone/blocks.{b-1}/Add_1_output_0'
    node_by_name[f'/model/backbone/blocks.{b}/norm1/LayerNormalization'].input[0] = prev_out
    node_by_name[f'/model/backbone/blocks.{b}/Add'].input[0] = prev_out

# 4. Feature extraction reshapes (blocks 4, 11, 17, 23)
feat_blocks = [(4, 'Reshape_15'), (11, 'Reshape_29'), (17, 'Reshape_41'), (23, 'Reshape_53')]
for b, r_name in feat_blocks:
    r_node = node_by_name[f'/model/backbone/{r_name}']
    r_node.input[0] = f'/model/backbone/blocks.{b}/Add_1_output_0'
    r_node.input[1] = 'da3_static_feat_norm_shape'

# The 4 cut points between Backbone and Head:
# We cut at the block outputs directly!
cut_tensors = [
    '/model/backbone/blocks.4/Add_1_output_0',
    '/model/backbone/blocks.11/Add_1_output_0',
    '/model/backbone/blocks.17/Add_1_output_0',
    '/model/backbone/blocks.23/Add_1_output_0',
]

cut_value_infos = [
    helper.make_tensor_value_info(cut_tensors[0], onnx.TensorProto.FLOAT, [1, 1297, 1024]),
    helper.make_tensor_value_info(cut_tensors[1], onnx.TensorProto.FLOAT, [1, 1297, 1024]),
    helper.make_tensor_value_info(cut_tensors[2], onnx.TensorProto.FLOAT, [1, 1297, 1024]),
    helper.make_tensor_value_info(cut_tensors[3], onnx.TensorProto.FLOAT, [1, 1297, 1024]),
]

# Backward trace for Head (from 'depth', 'sky' back to cut_tensors)
node_by_output = {}
for n in m.graph.node:
    for o in n.output:
        node_by_output[o] = n

inits_by_name = {i.name: i for i in m.graph.initializer}

head_nodes = []
head_inits = []
head_visited = set(cut_tensors)
queue = ['depth', 'sky']

while queue:
    curr = queue.pop(0)
    if curr in head_visited:
        continue
    head_visited.add(curr)
    
    if curr in inits_by_name:
        head_inits.append(inits_by_name[curr])
        continue
        
    if curr in node_by_output:
        n = node_by_output[curr]
        if n not in head_nodes:
            head_nodes.append(n)
            for inp in n.input:
                if inp and inp not in head_visited:
                    queue.append(inp)

head_node_set = set(id(n) for n in head_nodes)
head_nodes = [n for n in m.graph.node if id(n) in head_node_set]
head_inits = list({i.name: i for i in head_inits}.values())

head_graph = helper.make_graph(
    nodes=head_nodes,
    name='da3_head',
    inputs=cut_value_infos,
    outputs=list(m.graph.output),
    initializer=head_inits,
)
head_model = helper.make_model(head_graph, opset_imports=m.opset_import)
head_path = '/home/ubuntu/da3-npu/models/clean_head.onnx'
print(f"Saving clean head ({len(head_nodes)} nodes, {len(head_inits)} inits) to {head_path}...")
onnx.save(head_model, head_path)

# Backward trace for Backbone (from cut_tensors back to 'image')
bb_nodes = []
bb_inits = []
bb_visited = set(['image'])
queue = list(cut_tensors)

while queue:
    curr = queue.pop(0)
    if curr in bb_visited:
        continue
    bb_visited.add(curr)
    
    if curr in inits_by_name:
        bb_inits.append(inits_by_name[curr])
        continue
        
    if curr in node_by_output:
        n = node_by_output[curr]
        if n not in bb_nodes:
            bb_nodes.append(n)
            for inp in n.input:
                if inp and inp not in bb_visited:
                    queue.append(inp)

bb_node_set = set(id(n) for n in bb_nodes)
bb_nodes = [n for n in m.graph.node if id(n) in bb_node_set]
bb_inits = list({i.name: i for i in bb_inits}.values())

bb_graph = helper.make_graph(
    nodes=bb_nodes,
    name='da3_backbone',
    inputs=list(m.graph.input),
    outputs=cut_value_infos,
    initializer=bb_inits,
)
bb_model = helper.make_model(bb_graph, opset_imports=m.opset_import)
bb_path = '/home/ubuntu/da3-npu/models/clean_backbone.onnx'
print(f"Saving clean backbone ({len(bb_nodes)} nodes, {len(bb_inits)} inits) to {bb_path}...")
onnx.save(bb_model, bb_path)

print("Pipeline built successfully!")
