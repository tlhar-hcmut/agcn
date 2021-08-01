from src.main.feeder.ntu import NtuFeeder
from src.main.model import ParallelNet  
from src.main.config import cfg_ds
import torch

def load_data(cfg):
        
        _feeder_train = NtuFeeder(
            path_data="{}/train_{}_joint.npy".format(cfg.path_data_preprocess, cfg.benchmark),
            path_label="{}/train_{}_label.pkl".format(cfg.path_data_preprocess, cfg.benchmark),
            random_speed=True
        )

        _feeder_test = NtuFeeder(
            path_data="{}/val_{}_joint.npy".format(cfg.path_data_preprocess, cfg.benchmark),
            path_label="{}/val_{}_label.pkl".format(cfg.path_data_preprocess, cfg.benchmark),
            # random_speed=True
        )

        return {"train": _feeder_train, "val": _feeder_test}

if __name__=="__main__":
    #predict for parallel, 26 joints, xsub
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParallelNet(** cfg_ds.__dict__).to(device)
    data_feeder = load_data(cfg_ds) 

    data_np = torch.tensor(data_feeder["val"].data).to(device)
    label = torch.tensor(data_feeder["val"].label).to(device)
    sample_name = data_feeder["val"].sample_name
    predictions = model(data_np)
    pass