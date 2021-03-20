import unittest

from main.feeder.ntu import NtuFeeder


class TestFeeder(unittest.TestCase):
    def setUp(self) -> None:
        feeder = NtuFeeder(
            path_data="",
            path_label="",
            ls_class
        )

    def test(self):
        pass
