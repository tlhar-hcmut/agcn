import os
import unittest
from typing import *

import numpy as np
from src.main import generator
from src.main.util.config import config_glob


class TestGenerator(unittest.TestCase):
    def test_get_nonzero_std(self):
        data: np.ndarray = np.array([[[1, 2, 3]], [[3, 4, 5]], [[5, 6, 7]]])
        self.assertEqual(generator.get_nonzero_std(data), 4.898979485566356)

    def test_read_skeleton(self):
        dir_data: str = "/data/extracts/nturgb+d_skeletons"
        path_data: str = dir_data + "/S001C001P001R001A043.skeleton"
        skeleton: Dict = generator.read_skeleton(path_data)

        self.assertEqual(skeleton["numFrame"], 85)
        self.assertEqual(skeleton["frameInfo"][0]["numBody"], 1)
        self.assertEqual(skeleton["frameInfo"][0]["bodyInfo"][0]["numJoint"], 25)
        self.assertEqual(skeleton["numFrame"], len(skeleton["frameInfo"]))
        self.assertEqual(
            skeleton["frameInfo"][0]["numBody"],
            len(skeleton["frameInfo"][0]["bodyInfo"]),
        )
        self.assertEqual(
            skeleton["frameInfo"][0]["bodyInfo"][0]["numJoint"],
            len(skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"]),
        )
        self.assertEqual(
            skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"][0]["x"],
            0.327525,
        )
        self.assertEqual(
            skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"][0]["y"],
            0.1586938,
        )
        self.assertEqual(
            skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"][0]["z"],
            3.7817,
        )

    def test_read_xyz(self):
        dir_data: str = "/data/extracts/nturgb+d_skeletons"
        path_data: str = dir_data + "/S001C001P001R001A043.skeleton"
        data1: np.ndarray = generator.read_xyz(path_data, 4, 10)
        self.assertEqual(data1.shape, (3, 85, 10, 4))
        data2: np.ndarray = generator.read_xyz(path_data, 2, 25)
        self.assertEqual(data2.shape, (3, 85, 25, 2))
        self.assertEqual(data2[0][0][0][0], 0.327525)

    def test_gen_joint(self):
        for b in list(config_glob["benchmarks"].keys()):
            for p in config_glob["phases"]:
                out_path = os.path.join(config_glob["path_data_preprocess"], b, p)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

        generator.gen_joint(
            config_glob["path_data_raw"],
            config_glob["path_data_preprocess"],
            config_glob["path_data_ignore"],
            [1, 2],
            config_glob["num_frame"],
            config_glob["num_joint"],
            config_glob["num_body"],
            config_glob["max_body"],
            config_glob["benchmarks"],
        )
