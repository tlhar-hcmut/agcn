from typing import Dict
import torch
import torch.optim as optim
from src.main.config import cfg_ds_v1
from src.main.feeder.ntu import NtuFeeder
from src.main.graph import NtuGraph
from src.main.model.agcn import UnitAGCN
from torch import nn
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from src.main.config import cfg_ds_v1
from src.main.config import cfg_train_xview
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from  src.main.util import pprinter

class TrainXView:
    def __init__(self):
        
        self.num_of_epoch=30

        self.model = UnitAGCN(num_class=12, cls_graph=NtuGraph)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _feeder_train = NtuFeeder(
            path_data=cfg_ds_v1.path_data_preprocess+"/train_xview_joint.npy",
            path_label=cfg_ds_v1.path_data_preprocess+"/train_xview_label.pkl",
        )
        _loader_train = DataLoader(
            dataset=_feeder_train,
            batch_size=8,
            shuffle=False,
            num_workers=1,
        )
        _feeder_test = NtuFeeder(
            path_data=cfg_ds_v1.path_data_preprocess+"/train_xview_joint.npy",
            path_label=cfg_ds_v1.path_data_preprocess+"/train_xview_label.pkl",
        )
        _loader_test = DataLoader(
            dataset=_feeder_test,
            batch_size=8,
            shuffle=False,
            num_workers=1,
        )
        self.loader_data: Dict = {"train": _loader_train, "test": _loader_test}

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.004)

        self.loss = nn.CrossEntropyLoss()

        self.best_acc = {"value" : 0, "epoch": 0}

    def __load_to_device(self):
        
        self.model.to(self.device)

        self.loss.to(self.device)
   
    def __calculate_metric(self, full_predictions: torch.tensor, loader_name='test'):
        true_labels = torch.tensor(self.loader_data[loader_name].dataset.label).to(self.device)
        predict_labels = torch.argmax(full_predictions,1).to(self.device)
        pprinter.pp(title=true_labels.shapes)
        pprinter.pp(title=predict_labels.shape)
        hit_cases = true_labels == predict_labels
        return sum(hit_cases) * 1.0 / len(hit_cases)

    def evaluate(self, epoch, save_score=False, loader_name=['test'], fail_case_file=None, pass_case_file=None):
        if fail_case_file is not None:
            f_fail = open(fail_case_file, 'w')
        if pass_case_file is not None:
            f_pass = open(pass_case_file, 'w')
        
        #set mode children into eval(): dropout, batchnorm,..
        self.model.eval()
        pprinter.pp(title='Evaluate epoch: {}'.format(epoch))
        for ln in loader_name:
            loss_value_list = []
            output_batch_list = []
            step = 0
            process = tqdm(self.loader_data[ln])
            #_ is batch_idx
            for _, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().to(self.device)
                    label = label.long().to(self.device)
                    output = self.model(data)
                    
                    loss = self.loss(output, label)
                    output_batch_list.append(output.data)
                    loss_value_list.append(loss.data.item())

                    #get maximum in axes 1 (row): return max_values, max_indices
                    _, max_indices = torch.max(output.data, 1)
                    step += 1

                if fail_case_file is not None or pass_case_file is not None:
                    predictions = list(max_indices.cpu().numpy())
                    lables = list(label.data.cpu().numpy())
                    for i, x in enumerate(predictions):
                        if x == lables[i] and pass_case_file is not None:
                            f_pass.write(str(index[i]) + ',' + str(x) + ',' + str(lables[i]) + '\n')
                        if x != lables[i] and fail_case_file is not None:
                            f_fail.write(str(index[i]) + ',' + str(x) + ',' + str(lables[i]) + '\n')
            #one hot vector predictions
            full_outputs=torch.cat(output_batch_list, dim=0)
            full_loss = np.mean(loss_value_list)
            accuracy = self.__calculate_metric(full_outputs)
            if accuracy > self.best_acc["value"]:
                self.best_acc["value"] = accuracy
                self.best_acc["epoch"] = epoch

            
            pprinter.pp(title='loss: {} epoch: {}'.format(full_loss, epoch))
            pprinter.pp(title='acc: {} epoch: {}'.format(accuracy, epoch))
            
            #scores is the highest value of predictions in each row:
            scores = torch.max(full_outputs,1)
            score_dict = dict(zip(self.loader_data[ln].dataset.sample_name, scores))
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        "output_train", epoch, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def train(self):
        losses=[]
        self.__load_to_device()
        for epoch in range(1, self.num_of_epoch+1):
            losses_epoch=[]
            for _, (data, label, _) in enumerate(tqdm(self.loader_data["train"])):
                data = data.float().to(self.device)
                data.requires_grad=False
                label = label.long().to(self.device)
                label.requires_grad=False

                #forward
                output_batch = self.model(data)
                loss_batch = self.loss(output_batch, label)

                #backward
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()
                losses_epoch.append(loss_batch)
            losses.append(torch.mean(torch.tensor(losses_epoch, dtype=torch.float)))
            self.evaluate(epoch, save_score=True, loader_name=["train"], fail_case_file="output_train/result_fail.txt", pass_case_file="output_train/result_pass.txt")
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("output_train/losses{}.png".format(epoch))
        torch.save(self.model.state_dict(), "output_train/model.pt")


if __name__ == "__main__":
    trainxview = TrainXView()
    trainxview.train()
    pprinter.pp(title="The best accuracy: {}".format(trainxview.best_acc))
    
