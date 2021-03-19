import unittest

from src.main.generator import read_skeleton


class TestGenerator(unittest.TestCase):
    def test_read_skeleton(self):
        read_skeleton(
            "/data/extracts/raw_ntu_120/nturgb+d_skeletons/S001C001P001R001A043.skeleton"
        )
        pass
