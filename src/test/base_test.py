import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
import torch
from torchsummary import summary
import sys
from src.main.config import CfgTrainLocal
from xcommon import xfile
import pytz
from datetime import datetime


output_architecture = CfgTrainLocal.output_train
xfile.mkdir(output_architecture)

class BaseTestCase(unittest.TestCase):
    def summary_to_file(self, title ,**kargs):
        with open(CfgTrainLocal.output_train + "/architecture_test.txt", 'a') as f:
            sys.stdout = f
    
            print("\n\n--------------------\n", datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')),": ", title, "\n--------------------\n")
            summary(**kargs)
