from src.main.model import Net
from src.main.graph import NtuGraph
import torchviz
import torch

def draw_compu_graph(model):
    # model.backward()
    model.to("cuda")
    input = torch.ones((1,3,300,25,2), dtype=torch.float32).to("cuda")
    output = model(input)
    dot = torchviz.make_dot(output)
    dot.format = 'png'
    dot.render("computation_graph")

if __name__ == "__main__":
    model = Net("cuda", num_class=12, cls_graph=NtuGraph)
    draw_compu_graph(model)
