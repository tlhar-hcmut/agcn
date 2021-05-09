import src.main.model as M
from src.main.graph import NtuGraph

from .base_test import BaseTestCase
from src.main.config import CfgTrainLocal
import torch


class TestModelVisualize(BaseTestCase):
    # def test_tk_model(self):
    #     model = M.TKNet(num_class=12, cls_graph=NtuGraph)
    #     self.summary_to_file("TkModel", model.to("cuda"), (3, 300, 25, 2), depth=5)

    # def test_tk_spatial(self):
    #     model = M.TKNet(num_class=12, stream=[0], cls_graph=NtuGraph)
    #     self.summary_to_file("Spatial", model.to("cuda"), (3, 300, 25, 2))

    # def test_tk_temporal(self):
    #     model = M.TKNet(num_class=12, stream=[1], cls_graph=NtuGraph)
    #     self.summary_to_file("Temporal", model.to("cuda"), (3, 300, 25, 2))

    def test_tk_temporal(self):
        model = M.TKNet( input_size=CfgTrainLocal.input_size, 
                                stream=CfgTrainLocal.stream, 
                                num_class=12, 
                                cls_graph=NtuGraph, 
                                num_head=CfgTrainLocal.num_head, 
                                num_block=CfgTrainLocal.num_block,
                                dropout  =CfgTrainLocal.dropout,
                                len_feature_new = CfgTrainLocal.len_feature_new)
        self.summary_to_file("Temporal",model=model, depth=15, col_width=20, col_names=["input_size","kernel_size", "output_size", "num_params"], input_data=torch.empty((1,*CfgTrainLocal.input_size)))

