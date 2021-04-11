import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
from torchsummary import summary
import torch

class TestModel(unittest.TestCase):
    def test_agcn(self):
        # C, T, V, M = 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = M.Net(num_class=12, cls_graph=NtuGraph).to(device)
        summary(model, input_size=(3, 300, 25, 2))

    # def test_gcn(self):
    #     model = M.UnitGCN(3, 64, np.ones((25, 25)))

    # def test_tcn(self):
    #     model = M.UnitTCN(64, 64)

    # def test_tgcn(self):
    #     model = M.UnitTGCN(3, 64, np.ones((25, 25)))

if __name__ == "__main__":
    a = TestModel()
    a.test_agcn()