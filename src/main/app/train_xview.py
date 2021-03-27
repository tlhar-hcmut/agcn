import torch
from src.main.feeder.ntu import NtuFeeder
from src.main.graph import NtuGraph
from src.main.model.agcn import UnitAGCN
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from src.main.config import cfg_ds_v1
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cpu")
    
    feeder_train = NtuFeeder(
        path_data= cfg_ds_v1.path_data_preprocess+"/train_xview_joint.npy",
        path_label=cfg_ds_v1.path_data_preprocess+"/train_xview_label.pkl",
    )

    loader_train = DataLoader(
        dataset=feeder_train,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    loss = nn.CrossEntropyLoss().cuda("gpu")
    
    #declarations:
    model = UnitAGCN(num_class=12, cls_graph=NtuGraph)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.004)
    losses =[]
    model.train()
    for epoch in range(5):
        for batch_idx, (data, label, index) in enumerate(tqdm(loader_train)):
            data = Variable(data.float().to(device), requires_grad=False).cuda()
            label = Variable(label.long().to(device), requires_grad=False).cuda()
            #forward
            output_ = model(data)
            #convert label nominal into one-hot
            # label_onehot = torch.zeros((label.data.shape[0], 12))
            # row = torch.arange(0, label.data.shape[0], dtype=torch.long)
            # label_onehot[row, label.data] = 1
            loss_ = loss(output_, label)

            #backward
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            losses.append(loss_.data.item())
        
            if (len(losses)%100!=0): continue
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("output_train/losses{}.png".format(epoch))
    torch.save(model.state_dict(), "output_train/weight/model.pt")