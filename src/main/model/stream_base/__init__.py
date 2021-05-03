from torch import optim, nn
import torchmetrics
from pytorch_lightning import LightningModule
from src.main.util import logger


class StreamBase(LightningModule):
    def __init__(
        self,
        input_size=(3, 300, 25, 2),
        num_class=60,
        cls_graph=NtuGraph,
        graph_args=dict(),
    ):
        super(StreamBase, self).__init__()

        self.metric_acc = torchmetrics.Accuracy()
        self.logger_train = logger.setup_logger(name="train", log_file="./output/train/log.log")
        self.logger_val = logger.setup_logger(name="val", log_file="./output/val/log.log")

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {'loss' :nn.functional.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

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
        return {'loss' :nn.functional.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

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