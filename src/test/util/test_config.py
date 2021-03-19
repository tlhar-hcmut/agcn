import unittest

from src.main.util.config import config_glob


class TestConfig(unittest.TestCase):
    def testConfigGlob(self):
        print(config_glob)
