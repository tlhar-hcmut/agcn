import unittest
from typing import *

from src.main.util import analyst
from src.main.util.config import config_glob


class TestConfig(unittest.TestCase):
    def test_list_info(self):
        print(analyst.list_info(config_glob["path_data_raw"], [1, 2, 3]))

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
