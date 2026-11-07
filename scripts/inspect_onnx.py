"""Prints the node sequence of one GDN layer and one attention layer of the decoder graph."""
import collections, sys
import onnx
from onnx import numpy_helper

path = sys.argv[1]
m = onnx.load(path, load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}
def shape(n):
    i = inits.get(n)
    return tuple(i.dims) if i is not None else None
def attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT: out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT: out[a.name] = round(a.f, 6)
        elif a.type == onnx.AttributeProto.INTS: out[a.name] = list(a.ints)[:6]
        elif a.type == onnx.AttributeProto.STRING: out[a.name] = a.s.decode()
    return out
def dump(nodes, depth=0):
    for n in nodes:
        ins = [f"{x}{shape(x) or ''}" for x in n.input]
        print("  " * depth + f"{n.op_type:28} {n.name[-60:]:60} in={ins} out={list(n.output)} {attrs(n)}")
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH and depth < 1 and a.name == "then_branch":
                print("  " * depth + f"  [then_branch of {n.name}]")
                dump(a.g.node, depth + 2)
for prefix in sys.argv[2:]:
    print("=" * 30, prefix)
    dump([n for n in g.node if prefix in n.name])
print("=" * 30, "graph tail")
dump([n for n in g.node if "/model/layers." not in n.name][-25:])
