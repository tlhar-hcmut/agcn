import os

import torch
from src.main.config import cfg_ds
from src.main.util import pprinter
from src.main.feeder.ntu import NtuFeeder
from torch.utils.data import DataLoader
from src.main.util import pprinter


from typing import Dict




def _load_npy():
    _feeder_train = NtuFeeder(
        path_data="{}/train_{}_joint.npy".format(cfg_ds.path_data_preprocess, cfg_ds.benchmark),
        path_label="{}/train_{}_label.pkl".format(cfg_ds.path_data_preprocess, cfg_ds.benchmark),
    )
   
    _feeder_test = NtuFeeder(
        path_data="{}/val_{}_joint.npy".format(cfg_ds.path_data_preprocess, cfg_ds.benchmark),
        path_label="{}/val_{}_label.pkl".format(cfg_ds.path_data_preprocess, cfg_ds.benchmark),
    )

    loader_data: Dict = {"train": _feeder_train, "val": _feeder_test}
    return loader_data

if __name__ == "__main__":

    loader_data = _load_npy()
    pprinter.pp_scalar(None,"Train data")
    pprinter.pp_scalar(len(loader_data["train"].label))
    pprinter.pp_scalar(set(loader_data["train"].label))

    pprinter.pp_scalar(None, "Validation data")
    pprinter.pp_scalar(len(loader_data["val"].label))
    pprinter.pp_scalar(set(loader_data["val"].label))
#   for _, (ts_data_batch, ts_label_batch, index) in enumerate(process):

