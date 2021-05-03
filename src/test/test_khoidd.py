import unittest

import numpy as np
import src.main.model.stream_khoidd as M
import torch
from src.main.graph import NtuGraph
from torchsummary import summary


class TestModel(unittest.TestCase):
    def test_tk_model(self):
        model = M.KhoiddNet(num_class=12, cls_graph=NtuGraph)
        summary(model.to("cpu"), input_size=(3, 300, 25, 2))
        print(model.forward(torch.ones((1, 3, 300, 25, 2))).shape)


    def test_stream_spatial(self):
        model = M.StreamSpatialGCN(input_size=(3, 300, 25, 2), cls_graph=NtuGraph)
        summary(model.to("cpu"), input_size=(3, 300, 25, 2))
        print(model.forward(torch.ones((1, 3, 300, 25, 2))).shape)

    def test_stream_temporal(self):
        model = M.StreamTemporalGCN(input_size=(3, 300, 25, 2))
        summary(model.to("cpu"), input_size=(3, 300, 25, 2))
        print(model.forward(torch.ones((1, 3, 300, 25, 2))).shape)

