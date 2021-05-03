from functools import *

import src.main.config.cfg_train as cfg_train
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from src.main.util import logger
from torch import nn, optim
from xcommon import xfile

from .stream_spatial import *
from .stream_temporal import *


class StreamBase(LightningModule):
    def __init__(self, path_output="./output"):
        super(StreamBase, self).__init__()

        xfile.mkdir(f"{path_output}/train")
        xfile.mkdir(f"{path_output}/val")

        self.metric_acc = torchmetrics.Accuracy()
        self.logger_train = logger.setup_logger("train", f"{path_output}/train/log.log")
        self.logger_val = logger.setup_logger("val", f"{path_output}/val/log.log")

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {"loss": nn.functional.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs) -> None:
        self._validate(outputs)

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {"loss": nn.functional.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs) -> None:
        self._validate(outputs)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def _validate(self, outputs):
        loss = 0.0
        acc = 0.0
        for output in outputs:
            loss = loss + output["loss"].item()
            acc = acc + self.metric_acc(output["y_hat"].softmax(-1), output["y"])
        loss = loss / len(outputs)
        acc = acc / len(outputs)
        self.logger_val.info(f"loss: {loss} acc: {acc}")


class TKNet(StreamBase):
    def __init__(
        self,
        name="",
        path_output="./output",
        stream=[0, 1],
        input_size=(3, 300, 25, 2),
        num_class=12,
        cls_graph=NtuGraph,
        graph_args=dict(),
    ):
        super(TKNet, self).__init__(path_output=path_output)

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        self.input_size = input_size
        self.name = name

        self.stream_indices = stream

        self.streams = nn.ModuleList(
            [
                StreamSpatialGCN(input_size=input_size, cls_graph=cls_graph),
                StreamTemporalGCN(
                    input_size=input_size,
                    cls_graph=cls_graph,
                    dropout=cfg_train.dropout,
                    num_head=cfg_train.num_head,
                    num_block=cfg_train.num_block,
                    len_feature_new=cfg_train.len_feature_new,
                ),
            ]
        )

        num_stream_units = [64, 300]

        num_concat_units = sum(num_stream_units[i] for i in stream)

        self.fc1 = nn.Linear(num_concat_units, 64)

        self.fc2 = nn.Linear(64, num_class)

    def forward(self, x):

        output_streams = tuple(map(lambda i: self.streams[i](x), self.stream_indices))

        output_concat = torch.cat(output_streams, dim=1)

        return self.fc2(F.relu(self.fc1(output_concat)))
