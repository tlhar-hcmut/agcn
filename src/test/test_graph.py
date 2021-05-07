import unittest

import src.main.model as M
from torchsummary import summary


class TestGraph(unittest.TestCase):
    def test_stream_khoidd(self):
        model = M.KhoiDDNet(name="test")
        summary(model.to("cuda"), input_size=(3, 300, 25, 2))

