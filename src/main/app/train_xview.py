from typing import Dict

import torch
from src.main.graph import NtuGraph
from torch import nn
from src.main.model import TKNet
from src.main.app.base_train import BaseTrainer
from src.main.config import CfgTrainLocal



if __name__ == "__main__":

    trainer = BaseTrainer(
                    models=[
                        TKNet( input_size=CfgTrainLocal.input_size, 
                                stream=CfgTrainLocal.stream, 
                                num_class=12, 
                                cls_graph=NtuGraph, 
                                num_head=CfgTrainLocal.num_head, 
                                num_block=CfgTrainLocal.num_block,
                                dropout  =CfgTrainLocal.dropout,
                                len_feature_new = CfgTrainLocal.len_feature_new)],
                    cfgs=[
                        CfgTrainLocal()]
                )

    trainer.train()
        