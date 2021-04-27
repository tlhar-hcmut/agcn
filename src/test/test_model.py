import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
import torch
from torchsummary import summary


class TestModel(unittest.TestCase):
    def test_tk_model(self):
        model = M.TKNet(device=torch.device("cuda"), cls_graph=NtuGraph)
        summary(model.to('cuda'), input_size=(3, 300, 25, 2))

    def test_stream_temporal(self):
        model = M.StreamTemporalGCN(device=torch.device("cuda"), num_class=12, cls_graph=NtuGraph)
        summary(model.to('cuda'), input_size=(3, 300, 25, 2))

    def test_stream_spatial(self):
        model = M.StreamSpatialGCN(num_class=12, cls_graph=NtuGraph)
        summary(model.to('cuda'), input_size=(3, 300, 25, 2))

    def test_tagcn(self):
        model = M.UnitTAGCN(3, 64, np.ones((25, 25)))
        summary(model.to('cuda'), input_size=(3, 300, 25))

    def test_agcn(self):
        model = M.UnitAGCN(3, 64, np.ones((25, 25)))
        summary(model.to('cuda'), input_size=(3, 300, 25))

    def test_tcn(self):
        model = M.UnitTCN(3, 32)
        summary(model.to('cuda'), input_size=(3, 300, 25))

