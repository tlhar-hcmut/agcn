import unittest

import numpy as np
from src.main.generator import processor, read_xyz
from src.main.util import SkeletonType, draw_skeleton


class TestUtil(unittest.TestCase):
    def test_draw_skeleton(self):
        dir_data = "/data/extracts/nturgb+d_skeletons"
        path_data = dir_data + "/S001C001P001R001A043.skeleton"

        # draw raw data
        input_raw = read_xyz(path_data)
        draw_skeleton(input_raw, SkeletonType.RAW, "./output/")

        # draw preprocessed data
        input_preprocess = np.array(processor.normalize(np.expand_dims(input_raw, axis=0), silent=True))
        input_preprocess = np.array(np.squeeze(input_preprocess, axis=0))
        draw_skeleton(input_preprocess, SkeletonType.PREPROCESSED, "./output/", "S001C001P001R001A043")
