import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
import torch
from torchsummary import summary


class TestModel(unittest.TestCase):
    def test_tk_model(self):
        model = M.TKNet(num_class=12, cls_graph=NtuGraph)
        summary(model.to('cuda'), input_size=(3, 300, 25, 2))

    def test_tk_spatial(self):
        model = M.TKNet(num_class=12, stream=[0], cls_graph=NtuGraph)
        summary(model.to('cuda'), input_size=(3, 300, 25, 2))

    def test_tk_temporal(self):
        model = M.TKNet(num_class=12, stream=[1], cls_graph=NtuGraph)
        summary(model.to('cuda'), input_size=(3, 300, 25, 2))

    
