import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as metric
from pytorch_lightning import LightningModule
from torch import nn, optim

from src.main.graph import NtuGraph
from src.main.util import logger
from . import util
from .sgcn import UnitSpatialGcn
from .tgcn import UnitTemporalGcn


class KhoiddNet(LightningModule):
    def __init__(
        self,
        input_size=(3, 300, 25, 2),
        num_class=60,
        cls_graph=NtuGraph,
        graph_args=dict(),
    ):
        super(KhoiddNet, self).__init__()

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        self.metric_acc = metric.Accuracy()
        self.logger_train = logger.setup_logger(name="train", log_file="./output/train/log.log")
        self.logger_val = logger.setup_logger(name="val", log_file="./output/val/log.log")

        self.stream_spatial = StreamSpatialGCN(
            input_size=input_size, cls_graph=cls_graph
        )
        # self.stream_temporal = StreamTemporalGCN(input_size=input_size)

        self.fc = nn.Linear(32, num_class)
        # self.fc = nn.Linear(64, num_class)

    def forward(self, x):
        # return self.fc(torch.cat((self.stream_spatial(x), self.stream_temporal(x)), 1))
        return self.fc(self.stream_spatial(x))

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {'loss' :F.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs) -> None:
        loss = 0.0
        acc = 0.0
        for output in outputs:
            loss = loss + output['loss'].item()
            acc = acc + self.metric_acc(output['y_hat'].softmax(-1), output['y'])
        loss = loss / len(outputs)
        acc = acc / len(outputs)
        self.logger_train.info(f'loss: {loss} acc: {acc}')

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {'loss' :F.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs) -> None:
        loss = 0.0
        acc = 0.0
        for output in outputs:
            loss = loss + output['loss'].item()
            acc = acc + self.metric_acc(output['y_hat'].softmax(-1), output['y'])
        loss = loss / len(outputs)
        acc = acc / len(outputs)
        self.logger_val.info(f'loss: {loss} acc: {acc}')

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class StreamSpatialGCN(torch.nn.Module):
    def __init__(
        self, input_size, cls_graph=None, graph_args=dict(),
    ):
        super(StreamSpatialGCN, self).__init__()

        C, T, V, M = input_size
        num_person = M
        in_channels = C
        num_joint = V

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_joint)

        self.l1 = UnitSpatialGcn(3, 8, A, residual=False)
        self.l2 = UnitSpatialGcn(8, 8, A)
        self.l3 = UnitSpatialGcn(8, 8, A)
        self.l4 = UnitSpatialGcn(8, 8, A)
        self.l5 = UnitSpatialGcn(8, 16, A, stride=2)
        self.l6 = UnitSpatialGcn(16, 16, A)
        self.l7 = UnitSpatialGcn(16, 16, A)
        self.l8 = UnitSpatialGcn(16, 32, A, stride=2)
        self.l9 = UnitSpatialGcn(32, 32, A)
        self.l10 = UnitSpatialGcn(32, 32, A)

        util.init_bn(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.contiguous()
            .view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.contiguous().view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return x


class StreamTemporalGCN(torch.nn.Module):
    def __init__(self, input_size):
        super(StreamTemporalGCN, self).__init__()

        C, T, V, M = input_size
        num_person = M
        in_channels = C
        num_frame = T

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_frame)
        A = np.ones((T, T))

        self.l1 = UnitTemporalGcn(3, 8, A, residual=False)
        self.l2 = UnitTemporalGcn(8, 8, A)
        self.l3 = UnitTemporalGcn(8, 8, A)
        self.l4 = UnitTemporalGcn(8, 8, A)
        self.l5 = UnitTemporalGcn(8, 16, A, stride=2)
        self.l6 = UnitTemporalGcn(16, 16, A)
        self.l7 = UnitTemporalGcn(16, 16, A)
        self.l8 = UnitTemporalGcn(16, 32, A, stride=2)
        self.l9 = UnitTemporalGcn(32, 32, A)
        self.l10 = UnitTemporalGcn(32, 32, A)

        util.init_bn(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 2, 1, 3).contiguous().view(N, M * T * C, V)
        x = self.data_bn(x)
        x = (
            x.contiguous()
            .view(N, M, T, C, V)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, V, T)
        )

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.contiguous().view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return x
