import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
import torch
from torchsummary import summary
import sys
from xcommon import xfile
from .base_test import BaseTestCase

class TestModelVisualize(BaseTestCase):
    # def test_tk_model(self):
    #     model = M.TKNet(num_class=12, cls_graph=NtuGraph)
    #     self.summary_to_file("TkModel",model.to('cuda'), (3, 300, 25, 2), depth=5)

    # def test_tk_spatial(self):
    #     model = M.TKNet(num_class=12, stream=[0], cls_graph=NtuGraph)
    #     self.summary_to_file("Spatial",model.to('cuda'), (3, 300, 25, 2))

    def test_tk_temporal(self):
        model = M.TKNet(num_class=12, stream=[1], cls_graph=NtuGraph)
        self.summary_to_file(title="Temporal",model = model.to('cuda'), input_size = (3, 300, 25, 2), depth=7)
