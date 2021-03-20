import torch
from src.main.feeder.ntu import NtuFeeder
from src.main.graph import NtuGraph
from src.main.model.agcn import UnitAGCN
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm.std import tqdm

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feeder_train = NtuFeeder(
        path_data="/data/preprocess/nturgb+d_skeletons/train_xview_joint.npy",
        path_label="/data/preprocess/nturgb+d_skeletons/train_xview_label.pkl",
    )

    loader_train = DataLoader(
        dataset=feeder_train,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    feeder_test = NtuFeeder(
        path_data="/data/preprocess/nturgb+d_skeletons/val_xview_joint.npy",
        path_label="/data/preprocess/nturgb+d_skeletons/val_xview_label.pkl",
    )

    loader_test = DataLoader(
        dataset=feeder_test,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    loss = nn.CrossEntropyLoss()

    model = UnitAGCN(num_class=12, cls_graph=NtuGraph)

    for batch_idx, (data, label, index) in enumerate(tqdm(loader_test)):
        data = Variable(data.float().to(device), requires_grad=False)
        label = Variable(label.float().to(device), requires_grad=False)

        output = model(data)
        # break
