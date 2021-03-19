import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph


class TestGenerator(unittest.TestCase):
    def test_agcn(self):
        model = M.UnitAGCN(cls_graph=NtuGraph)

    def test_gcn(self):
        model = M.UnitGCN(3, 64, np.ones((25, 25)))

    def test_tcn(self):
        model = M.UnitTCN(64, 64)

    def test_tgcn(self):
        model = M.UnitTGCN(3, 64, np.ones((25, 25)))
