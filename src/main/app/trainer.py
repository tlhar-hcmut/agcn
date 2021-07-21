from typing import Dict

import torch
from src.main.graph import NtuGraph
from torch import nn
from src.main.model import *
from src.main.app.base_train import BaseTrainer
from src.main.config import *



if __name__ == "__main__":

    trainer = BaseTrainer(
                    cls_models=[
                        # SequentialNet,
                        # ParallelNet,
                        AuthorNet
                        ]
                       )

    trainer.train()
