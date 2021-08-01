from src.main.feeder.ntu import NtuFeeder
from torch.utils.data import DataLoader
from src.main.model import ParallelNet  
from src.main.config import cfg_ds
import torch
from tqdm.std import tqdm
import os


def load_data(cfg):
        
        _feeder_train = NtuFeeder(
            path_data="{}/train_{}_joint.npy".format(cfg.path_data_preprocess, cfg.benchmark),
            path_label="{}/train_{}_label.pkl".format(cfg.path_data_preprocess, cfg.benchmark),
            random_speed=True
        )
        _loader_train = DataLoader(
            dataset=_feeder_train,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
        )
        _feeder_test = NtuFeeder(
            path_data="{}/val_{}_joint.npy".format(cfg.path_data_preprocess, cfg.benchmark),
            path_label="{}/val_{}_label.pkl".format(cfg.path_data_preprocess, cfg.benchmark),
            # random_speed=True
        )
        _loader_test = DataLoader(
            dataset=_feeder_test,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
        )
        return {"train": _loader_train, "val": _loader_test}


if __name__=="__main__":
    #predict for parallel, 26 joints, xsub
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParallelNet(** cfg_ds.__dict__).to(device)
    model.load_state_dict(torch.load(cfg_ds.pretrained_path))
    model.eval()
    data_loader = load_data(cfg_ds) 

    sample_name = data_loader["val"].dataset.sample_name
    labels = torch.tensor(data_loader["val"].dataset.label).to(device)
    predictions=[]

    for (data,label, _) in tqdm(data_loader["val"]):
        data = data.float().to(device)
        prediction = model(data)
        prediction = prediction.data
        predictions.append(prediction)
    predictions_ts = torch.cat(predictions, dim=0)
    
    output = torch.argmax(predictions_ts, dim=-1)
    samples_true_mark = output==labels

    os.makedirs(cfg_ds.output_train+'/predictions', exist_ok=True)
    with open(cfg_ds.output_train+'/predictions/fail_case.txt', 'a+') as f:
        for i, name in enumerate(sample_name):
            if samples_true_mark[i]:
                pass
            else:
                #Fail cases
                f.write(name)
                f.write('\n')