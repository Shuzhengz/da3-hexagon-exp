import onnx
from onnx import helper, numpy_helper

print("Loading model_fp32_nomatconv.onnx...")
m = onnx.load('/home/ubuntu/da3-npu/models/model_fp32_nomatconv.onnx', load_external_data=False)

feat_names = [
    '/model/head/Reshape_4_output_0',
    '/model/head/Reshape_5_output_0',
    '/model/head/Reshape_6_output_0',
    '/model/head/Reshape_7_output_0',
]

node_by_output = {}
for n in m.graph.node:
    for o in n.output:
        node_by_output[o] = n

inits_by_name = {i.name: i for i in m.graph.initializer}

# 1. Backward trace for Head (from graph outputs 'depth', 'sky' back to feat_names)
head_nodes = []
head_inits = []
head_visited = set(feat_names)
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

# Reorder head nodes to match topological order in original model
head_node_set = set(id(n) for n in head_nodes)
head_nodes = [n for n in m.graph.node if id(n) in head_node_set]
head_inits = list({i.name: i for i in head_inits}.values())

feat_value_infos = [
    helper.make_tensor_value_info(feat_names[0], onnx.TensorProto.FLOAT, [1, 1024, 36, 36]),
    helper.make_tensor_value_info(feat_names[1], onnx.TensorProto.FLOAT, [1, 1024, 36, 36]),
    helper.make_tensor_value_info(feat_names[2], onnx.TensorProto.FLOAT, [1, 1024, 36, 36]),
    helper.make_tensor_value_info(feat_names[3], onnx.TensorProto.FLOAT, [1, 1024, 36, 36]),
]

head_graph = helper.make_graph(
    nodes=head_nodes,
    name='da3_head',
    inputs=feat_value_infos,
    outputs=list(m.graph.output),
    initializer=head_inits,
)
head_model = helper.make_model(head_graph, opset_imports=m.opset_import)
head_path = '/home/ubuntu/da3-npu/models/head.onnx'
print(f"Saving head ({len(head_nodes)} nodes, {len(head_inits)} inits) to {head_path}...")
onnx.save(head_model, head_path)

# 2. Backward trace for Backbone (from feat_names back to 'image')
bb_nodes = []
bb_inits = []
bb_visited = set(['image'])
queue = list(feat_names)

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
    outputs=feat_value_infos,
    initializer=bb_inits,
)
bb_model = helper.make_model(bb_graph, opset_imports=m.opset_import)
bb_path = '/home/ubuntu/da3-npu/models/backbone.onnx'
print(f"Saving backbone ({len(bb_nodes)} nodes, {len(bb_inits)} inits) to {bb_path}...")
onnx.save(bb_model, bb_path)

print("Split completed successfully!")
