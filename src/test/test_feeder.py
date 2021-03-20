import unittest

from src.main.feeder.ntu import NtuFeeder


class TestFeeder(unittest.TestCase):
    def setUp(self) -> None:
        dir_ds = "/data/preprocess/nturgb+d_skeletons"
        self.feeder = NtuFeeder(
            path_data=dir_ds + "/train_xsub_joint.npy",
            path_label=dir_ds + "/train_xsub_label.pkl",
        )

    def test_getitem(self):
        self.assertEqual(self.feeder.__getitem__(0)[0].shape, (3, 300, 25, 2))
