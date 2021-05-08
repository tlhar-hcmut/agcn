import sys
import logging
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from src.main.graph import NtuGraph
from src.main.util import plot_confusion_matrix, setup_logger
from torch import nn
from tqdm.std import tqdm
from xcommon import xfile
from src.main.model import TKNet

from src.main.app.base_train import BaseTrainer
from src.main.config import CfgTrainLocal



if __name__ == "__main__":

    trainer = BaseTrainer()
    trainer.models=[
        TKNet(  input_size=CfgTrainLocal.input_size, 
                stream=CfgTrainLocal.stream, 
                num_class=12, 
                cls_graph=NtuGraph, 
                num_head=CfgTrainLocal.num_head, 
                num_block=CfgTrainLocal.num_block,
                dropout  =CfgTrainLocal.dropout,
                len_feature_new = CfgTrainLocal.len_feature_new)
                ]
    trainer.cfgs=[
        CfgTrainLocal()]
    trainer.train()
        