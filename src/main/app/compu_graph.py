import warnings
from src.main.model import *
from src.main.graph import NtuGraph
import torchviz
import torch
from src.main.config import *
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import time


def draw_compu_graph(model, input, format):
    # model.backward()
    model.to("cuda")
    output = model(input)
    dot = torchviz.make_dot(output)
    dot.format = format
    dot.render("computation_graph")





def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_vanishing_grad(grad_output):
        if grad_output is None:
            return False
        return (grad_output.abs().sum()<0.0000001)
    def is_explore_grad(grad_output):
        if grad_output is None:
            return False
        return (grad_output.abs() >= 1e4).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                # assert fn in fn_dict, fn
                if fn not in fn_dict:
                    print("fn not in fn_dict")
                    return
                fillcolor = 'white' 
                if any(is_vanishing_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                if any(is_explore_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'blue'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot



if __name__ == "__main__":
    # model = TKNet(**cfgTrainLocalMultihead1.__dict__).to("cuda")
    # model = stream_temporal.StreamTemporalGCN(**cfgTrainLocalMultihead1.__dict__).to("cuda")
    model = SequentialNet(**cfgTrainSequential_changenorm.__dict__).to("cuda")
    input  = torch.randn(1, 3, 300, 25, 2, requires_grad=True).to("cuda")

    draw_compu_graph(model, input, 'png')

    output = model(input).to("cuda") 
    # loss = nn.CrossEntropyLoss()
    # z = loss(output, torch.zeros((1, 1), dtype=torch.int64).to("cuda"))

    label  = torch.zeros((300,1)).to('cuda')
    # label[0][0]=1
    z = (output - label).sum()*10
    # z = output.sum()

    get_dot = register_hooks(z)
    
    z.backward()

    dot = get_dot()
    dot.format = 'jpg'
    dot.render("computation_graph_Track12")


