import torchmetrics
from pytorch_lightning import LightningModule
from src.main.util import logger
from torch import nn, optim
from xcommon import xfile


class StreamBase(LightningModule):
    def __init__(self, path_output="./output"):
        super(StreamBase, self).__init__()

        xfile.mkdir(f"{path_output}/train")
        xfile.mkdir(f"{path_output}/val")

        self.metric_acc = torchmetrics.Accuracy()
        self.logger_train = logger.setup_logger(
            name="train", log_file=f"{path_output}/train/log.log"
        )
        self.logger_val = logger.setup_logger(
            name="val", log_file=f"{path_output}/val/log.log"
        )

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {"loss": nn.functional.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs) -> None:
        loss = 0.0
        acc = 0.0
        for output in outputs:
            loss = loss + output["loss"].item()
            acc = acc + self.metric_acc(output["y_hat"].softmax(-1), output["y"])
        loss = loss / len(outputs)
        acc = acc / len(outputs)
        self.logger_train.info(f"loss: {loss} acc: {acc}")

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self(x)
        return {"loss": nn.functional.cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs) -> None:
        loss = 0.0
        acc = 0.0
        for output in outputs:
            loss = loss + output["loss"].item()
            acc = acc + self.metric_acc(output["y_hat"].softmax(-1), output["y"])
        loss = loss / len(outputs)
        acc = acc / len(outputs)
        self.logger_val.info(f"loss: {loss} acc: {acc}")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
