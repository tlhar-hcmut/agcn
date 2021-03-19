import unittest
from typing import *

from src.main.generator import read_skeleton


class TestGenerator(unittest.TestCase):
    def test_read_skeleton(self):
        dir_data: str = "/data/extracts/nturgb+d_skeletons"
        path_data: str = dir_data + "/S001C001P001R001A043.skeleton"
        skeleton: Dict = read_skeleton(path_data)

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
