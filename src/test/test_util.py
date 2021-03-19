import math
import unittest

import numpy as np
from src.main.util import analyst, rotation


class TestUtil(unittest.TestCase):
    def test_check_benchmark(self):
        dir_data: str = "/data/extracts/nturgb+d_skeletons"
        path_data1: str = dir_data + "/S001C001P001R001A043.skeleton"
        path_data2: str = dir_data + "/S001C001P001R001A042.skeleton"
        benmarks = {
            "setup_number": [1],
            "camera_id": [1],
            "performer_id": [1],
            "replication_number": [1],
            "action_class": [43],
        }
        self.assertEqual(analyst.check_benchmark(path_data1, benmarks), True)
        self.assertEqual(analyst.check_benchmark(path_data2, benmarks), False)

    def test_get_angle_between(self):
        vec_x: np.ndarray = np.array([1, 0, 0])
        vec_y: np.ndarray = np.array([0, 1, 0])
        vec_z: np.ndarray = np.array([0, 0, 1])
        self.assertEqual(rotation.get_angle_between(vec_x, vec_y), math.pi / 2)
        self.assertEqual(rotation.get_angle_between(vec_z, vec_y), math.pi / 2)
        self.assertEqual(rotation.get_angle_between(vec_z, vec_x), math.pi / 2)

    def test_cal_unit_vec(self):
        vec: np.ndarray = rotation.cal_unit_vec(np.random.randint(1, 10, size=(10, 1)))
        self.assertEqual(math.ceil(np.linalg.norm(vec)), 1)

    def test_rotate_matrix(self):
        vec_z: np.ndarray = np.array([0, 0, 1])
        mat_rot: np.ndarray = rotation.rotate_matrix(vec_z, math.pi).astype(np.int)
        mat_opt: np.ndarray = np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
        self.assertTrue(np.alltrue(mat_rot == mat_opt))
