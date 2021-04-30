import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
import torch
from torchsummary import summary
import sys
from src.main.config import cfg_train
from xcommon import xfile
import datetime


output_architecture = cfg_train.output_train
xfile.mkdir(output_architecture)

class BaseTestCase(unittest.TestCase):
    def summary_to_file(self, title ,**kargs):
        with open(cfg_train.output_train + "/architecture_test1.txt", 'a') as f:
            sys.stdout = f
    
            print("\n\n--------------------\n", datetime.datetime.now(),": ", title, "\n--------------------\n")
            summary(**kargs)
