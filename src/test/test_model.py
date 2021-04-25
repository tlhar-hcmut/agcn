import unittest

import numpy as np
import src.main.model as M
import torch
from src.main.graph import NtuGraph
from torchsummary import summary


class TestModel(unittest.TestCase):
    def test_tagcn(self):
        # C, T, V, M =
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = M.StreamSpatialGCN(num_class=12, cls_graph=NtuGraph).to(device)
        summary(model, input_size=(3, 300, 25, 2))

    def test_agcn(self):
        model = M.UnitAGCN(3, 64, np.ones((25, 25)))
        summary(model, input_size=(3, 300, 25, 2))

    def test_tcn(self):
        model = M.UnitTCN(64, 64)
        summary(model, input_size=(3, 300, 25, 2))

