import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
import torch
from torchsummary import summary
import sys
from src.main.config import cfg_train
from xcommon import xfile


output_architecture = cfg_train.output_train +"/architecture"
xfile.mkdir(output_architecture)

class BaseTestCase(unittest.TestCase):
    def summary_to_file(self, title ,*args):
        with open(cfg_train.output_train + "/architecture.txt", 'a') as f:
            sys.stdout = f
            print("\n\n--------------------\n",title, "\n--------------------\n")
            summary(*args)
