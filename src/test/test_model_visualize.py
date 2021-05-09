import src.main.model as M
from src.main.graph import NtuGraph

from .base_test import BaseTestCase
import src.main.config.cfg_train as cfg_train
import torch



class TestModelVisualize(BaseTestCase):
    def test_tk_model(self):
        model = M.TKNet(num_class=12, cls_graph=NtuGraph)
        self.summary_to_file("TkModel", model.to("cuda"), (3, 300, 25, 2), depth=5)

    def test_tk_spatial(self):
        model = M.TKNet(num_class=12, stream=[0], cls_graph=NtuGraph)
        self.summary_to_file("Spatial", model.to("cuda"), (3, 300, 25, 2))

    # def test_tk_temporal(self):
    #     model = M.TKNet(num_class=12, stream=[1], cls_graph=NtuGraph)
    #     self.summary_to_file("Temporal", model.to("cuda"), (3, 300, 25, 2))

    def test_tk_temporal(self):
        model = M.TKNet(num_class=12, stream=[1], input_size=cfg_train.input_size, cls_graph=NtuGraph)
        self.summary_to_file(title="Temporal", model=model.to('cuda'), input_data=torch.zeros((1, *cfg_train.input_size)), depth=15, col_width=20, col_names=["input_size", "kernel_size", "output_size", "num_params"])
