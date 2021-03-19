import unittest
from typing import Dict

import numpy as np
from src.main import generator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.dir_data = "/data/extracts/nturgb+d_skeletons"
        self.path_data = self.dir_data + "/S001C001P001R001A043.skeleton"

    def test_get_nonzero_std(self):
        data: np.ndarray = np.array([[[1, 2, 3]], [[3, 4, 5]], [[5, 6, 7]]])
        self.assertEqual(generator.get_nonzero_std(data), 4.898979485566356)

    def test_read_skeleton(self):
        skeleton: Dict = generator.read_skeleton(self.path_data)

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
            skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"][0]["x"], 0.327525,
        )
        self.assertEqual(
            skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"][0]["y"], 0.1586938,
        )
        self.assertEqual(
            skeleton["frameInfo"][0]["bodyInfo"][0]["jointInfo"][0]["z"], 3.7817,
        )

    def test_read_xyz(self):
        data1: np.ndarray = generator.read_xyz(self.path_data, 4, 10)
        self.assertEqual(data1.shape, (3, 85, 10, 4))
        data2: np.ndarray = generator.read_xyz(self.path_data, 2, 25)
        self.assertEqual(data2.shape, (3, 85, 25, 2))
        self.assertEqual(data2[0][0][0][0], 0.327525)

    def test_pad_null_frame(self):
        data: np.ndarray = generator.read_xyz(self.path_data, 2, 25)
        data: np.ndarray = np.expand_dims(data, axis=0)
        data: np.ndarray = np.transpose(data, [0, 4, 2, 3, 1])
        _, M, T, V, C = data.shape

        for num_frame_null in range(T):
            data[0, :M, T - num_frame_null : T, :V, :C] = 0
            generator.pad_null_frame(data, silient=True)
            frame_first = data[0, :M, T - num_frame_null : T, :V, :C].sum()
            frame_last = data[0, :M, 0:num_frame_null, :V, :C].sum()
            self.assertEqual(frame_first, frame_last)

    def test_sub_center_joint(self):
        data: np.ndarray = generator.read_xyz(self.path_data, 2, 25)
        data: np.ndarray = np.expand_dims(data, axis=0)
        data: np.ndarray = np.transpose(data, [0, 4, 2, 3, 1])
        _, M, T, V, C = data.shape
        body_center_old = data[0][0][:, 1:2, :].copy()
        body_part_old = data[0][0][:, 2:3, :].copy()

        generator.sub_center_joint(data)
        body_center_new = data[0][0][:, 1:2, :].copy()
        body_part_new = data[0][0][:, 2:3, :].copy()
        self.assertEqual(body_center_new.sum(), 0)
        self.assertEqual((body_part_new + body_center_old).sum(), body_part_old.sum())