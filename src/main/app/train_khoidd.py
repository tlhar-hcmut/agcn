from typing import Dict

import torch
import torch.optim as optim
from src.main.config import cfg_ds_v1
from src.main.feeder.ntu import NtuFeeder
from src.main.graph import NtuGraph
from src.main.model.stream_khoidd import TKNet
from torch import nn
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from xcommon import xfile, xlog


class TrainKhoidd:
    def __init__(self, num_of_epoch=40, path_pretrain=None):
        self.logger = xlog.get()
        self.model = TKNet(num_class=12, cls_graph=NtuGraph)
        self.num_of_epoch = num_of_epoch
        if path_pretrain != None:
            self.model.load_state_dict(torch.load(path_pretrain))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.CrossEntropyLoss()

        self.map_loader: Dict = {
            "train": DataLoader(
                dataset=NtuFeeder(
                    path_data=cfg_ds_v1.path_data_preprocess + "/val_xsub_joint.npy",
                    path_label=cfg_ds_v1.path_data_preprocess + "/val_xsub_label.pkl",
                    random_shift=True,
                ),
                batch_size=24,
            ),
            # "val": DataLoader(
            #     dataset=NtuFeeder(
            #         path_data=cfg_ds_v1.path_data_preprocess + "/train_xsub_joint.npy",
            #         path_label=cfg_ds_v1.path_data_preprocess + "/train_xsub_label.pkl",
            #     ),
            #     batch_size=24,
            #     shuffle=False,
            #     num_workers=2,
            # ),
        }
        xfile.mkdir("output")

    def train(self):
        self.logger.info("--------------------- start ---------------------")
        self.model.to(self.device)
        self.loss.to(self.device)

        try:
            size_ds = len(self.map_loader["train"].dataset)
            for idx_epoch in range(1, self.num_of_epoch + 1):
                acc = 0
                loss = 0
                for data, label, _ in tqdm(self.map_loader["train"]):
                    data = data.float().to(self.device)
                    data.requires_grad = False
                    label = label.long().to(self.device)
                    label.requires_grad = False

                    oupt = self.model(data)
                    grad = self.loss(self.model(data), label)

                    self.optimizer.zero_grad()
                    grad.backward()
                    self.optimizer.step()

                    acc += (torch.argmax(oupt, 1) == label).float().sum()
                    loss += grad.data.item()

                self.logger.info(f"acc: {acc/size_ds} loss: {loss} epoch: {idx_epoch}")
        finally:
            pass
            # for _, (data, label, index) in enumerate(self.map_loader["val"]):
            #     with torch.no_grad():
            #         data = data.float().to(self.device)
            #         label = label.long().to(self.device)
            #         output = self.model(data)
            #         util.check_fail(label, output, "output/val_fail_case.txt")


if __name__ == "__main__":
    TrainKhoidd().train()
