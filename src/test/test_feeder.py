import unittest

import numpy as np
from src.main.feeder.ntu import NtuFeeder
from torch.utils import data


class TestFeeder(unittest.TestCase):
    def setUp(self) -> None:
        dir_ds = "/data/preprocess/nturgb+d_skeletons_reorder"
        self.feeder_train = NtuFeeder(
            path_data=dir_ds + "/train_xsub_joint.npy",
            path_label=dir_ds + "/train_xsub_label.pkl",
            random_shift=True,
            random_speed=True,
        )

        self.feeder_val = NtuFeeder(
            path_data=dir_ds + "/val_xsub_joint.npy",
            path_label=dir_ds + "/val_xsub_label.pkl",
            random_shift=True,
            random_speed=True,
        )

    def test_size_ds(self):
        print(len(self.feeder_train))
        print(len(self.feeder_val))

    def test_getitem(self):
        self.assertEqual(self.feeder_train.__getitem__(0)[0].shape, (3, 300, 25, 2))

    def test_getitem(self):
        data_numpy = self.feeder_train.__getitem__(0)[0]
        data_numpy_old = np.copy(data_numpy)
        speed: int = 2
        indices = np.floor(np.arange(0, 300 * speed, speed)).astype(np.int)
        max_index = np.floor(300 / speed).astype(np.int)
        indices[max_index:] = 0
        data_numpy[:, :, :, :] = data_numpy[:, indices, :, :]
        data_numpy[:, max_index:, :, :] = 0

        self.assertEqual((data_numpy[:, 0, :, :] == data_numpy_old[:, 0, :, :]).all(), True)
        self.assertEqual((data_numpy[:, 1, :, :] == data_numpy_old[:, 2, :, :]).all(), True)
        self.assertEqual((data_numpy[:, 2, :, :] == data_numpy_old[:, 4, :, :]).all(), True)
        self.assertEqual((data_numpy[:, 1, :, :] == data_numpy_old[:, 1, :, :]).sum(), 3)
        self.assertEqual((data_numpy[:, 10, :, :] == data_numpy_old[:, 10, :, :]).sum(), 3)

