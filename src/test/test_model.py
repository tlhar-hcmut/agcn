import unittest

import numpy as np
import torch
import src.main.model as M
from src.main.graph import NtuGraph
from torchsummary import summary


class TestModel(unittest.TestCase):
    def test_tk_model(self):
        model = M.TKNet(cls_graph=NtuGraph)
        summary(model.to("cuda"), input_size=(3, 300, 25, 2))

    def test_stream_temporal(self):
        model = M.StreamTemporalGCN(input_size_temporal=(3, 300, 25, 2), cls_graph=NtuGraph)
        summary(model.to("cuda"), input_size=(3, 300, 25, 2))

    def test_stream_spatial(self):
        model = M.StreamSpatialGCN(pre_train=False)
        print(model.to("cpu").forward(torch.ones((1, 3, 300, 25, 2))))

    def test_tagcn(self):
        model = M.UnitTAGCN(3, 64, np.ones((25, 25)))
        summary(model.to("cuda"), input_size=(3, 300, 25))

    def test_agcn(self):
        model = M.UnitAGCN(3, 64, np.ones((25, 25)))
        summary(model.to("cuda"), input_size=(3, 300, 25))

    def test_tcn(self):
        model = M.UnitTCN(3, 32)
        summary(model.to("cuda"), input_size=(3, 300, 25))

    def test_self_attention(self):
        model = M.SelfAttention(16, 32)
        summary(model.to("cuda"), input_size=(300, 16))

    def test_multi_attention(self):
        model = M.MultiHeadAttention(1, 300, 16, 32)
        summary(model.to("cuda"), input_size=(300, 16))

    def test_stream_khoidd(self):
        model = M.StreamKhoiddGCN(name="test").to("cpu")
        print(model.forward(torch.ones((1, 3, 300, 25, 2))))
        summary(model.to("cpu"), input_size=(3, 300, 25, 2))
