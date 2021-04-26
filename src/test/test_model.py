import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
from torchsummary import summary


class TestModel(unittest.TestCase):
    def test_tk_model(self):
        model = M.TKNet(cls_graph=NtuGraph)
        summary(model, input_size=(3, 300, 25, 2))

    def test_stream_temporal(self):
        model = M.StreamTemporalGCN(25, 3)
        summary(model, input_size=(3, 300, 25, 2))

    def test_stream_spatial(self):
        model = M.StreamSpatialGCN(num_class=12, cls_graph=NtuGraph)
        summary(model, input_size=(3, 300, 25, 2))

    def test_tagcn(self):
        model = M.UnitTAGCN(3, 64, np.ones((64, 64)))
        summary(model, input_size=(3, 300, 25, 2))

    def test_agcn(self):
        model = M.UnitAGCN(3, 64, np.ones((25, 25)))
        summary(model, input_size=(3, 300, 25, 2))

    def test_tcn(self):
        model = M.UnitTCN(64, 64)
        summary(model, input_size=(3, 300, 25, 2))

