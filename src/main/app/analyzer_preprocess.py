import os

import torch
from src.main.config import cfg_ds
from src.main.util import pprinter
from src.main.feeder.ntu import NtuFeeder
from torch.utils.data import DataLoader
from src.main.util import pprinter

import pickle

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict




def _load_npy(benchmark):
    path_label_train="{}/train_{}_label.pkl".format(cfg_ds.path_data_preprocess, benchmark)
    path_label_val="{}/val_{}_label.pkl".format(cfg_ds.path_data_preprocess, benchmark)

    loader_data: Dict = {"train": load_data(path_label_train), "val": load_data(path_label_val)}
    return loader_data

def load_data(path_label):
    # data: N C T V M

    try:
        with open(path_label) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(path_label, "rb") as f:
            sample_name, label = pickle.load(f, encoding="latin1")
    return label


if __name__ == "__main__":

    for bm in cfg_ds.ls_benmark:
        benchmark=bm.name
        pprinter.pp_scalar(None,"Train data"+benchmark)
        loader_data = _load_npy(benchmark)
        pprinter.pp_scalar(None,"Train data")
        pprinter.pp_scalar(len(loader_data["train"]))
        pprinter.pp_scalar(set(loader_data["train"]))

        pprinter.pp_scalar(None, "Validation data")
        pprinter.pp_scalar(len(loader_data["val"]))
        pprinter.pp_scalar(set(loader_data["val"]))
    #   for _, (ts_data_batch, ts_label_batch, index) in enumerate(process):

        classes_train = [0.]*cfg_ds.num_class
        for x in loader_data["train"]:
            classes_train[x]+=1.0
        
        classes_val = [0.]*cfg_ds.num_class
        for x in loader_data["val"]:
            classes_val[x]+=1.0

        plt.xlabel("Class")
        plt.ylabel("The number of samples")
        opacity = 0.5
        bar_width = 0.35
        plt.xticks(range(len(classes_train)),tuple(range(0, len(classes_train))))
        bar1 = plt.bar(np.arange(len(classes_train)), classes_train, bar_width, align='center', alpha=opacity, color='b', label='Train')
        bar2 = plt.bar(np.arange(len(classes_val))+bar_width, classes_val, bar_width, align='center', alpha=opacity, color='r', label='Validation')
        
        # Add counts above the two bar graphs
        for rect in bar1 + bar2:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')

        plt.legend()
        plt.tight_layout()
        os.makedirs(cfg_ds.path_data_preprocess+"/ds_viz/", exist_ok=True)
        plt.savefig(cfg_ds.path_data_preprocess+"/ds_viz/"+benchmark+"_dataviz.png")
        plt.close()

