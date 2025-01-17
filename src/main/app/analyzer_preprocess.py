import os

import torch
from src.main.config import cfg_ds_v1
from src.main.util import pprinter
from src.main.feeder.ntu import NtuFeeder
from torch.utils.data import DataLoader
from src.main.util import pprinter


from typing import Dict




def _load_npy():
    _feeder_train = NtuFeeder(
        path_data=cfg_ds_v1.path_data_preprocess+"/val_xview_joint.npy",
        path_label=cfg_ds_v1.path_data_preprocess+"/val_xview_label.pkl",
    )
   
    _feeder_test = NtuFeeder(
        path_data=cfg_ds_v1.path_data_preprocess+"/train_xview_joint.npy",
        path_label=cfg_ds_v1.path_data_preprocess+"/train_xview_label.pkl",
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

