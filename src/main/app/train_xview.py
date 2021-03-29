from typing import Dict
import torch
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

class TrainXView:
    def __init__(self):
        # self.config=
        self.num_of_epoch=5

        self.model = UnitAGCN(num_class=12, cls_graph=NtuGraph)

        self.device = torch.device("cuda" if torch.cud.is_available() else "cpu")

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
        
        self.model().to(self.device)

        self.optimizer.to(self.device)

        self.loss.to(self.device)
   
    def __calculate_metric(self, full_predictions: torch.tensor, loader_name=['test']):
        true_labels = torch.tensor(self.loader_data[loader_name].dataset.label)
        predict_labels = torch.tensor(torch.argmax(full_predictions,1))
        print(true_labels.shape)
        print(predict_labels.shape)
        hit_cases = true_labels == predict_labels
        return sum(hit_cases) * 1.0 / len(hit_cases)

    def evaluate(self, epoch, save_score=False, loader_name=['test'], fail_case_file=None, pass_case_file=None):
        if fail_case_file is not None:
            f_fail = open(fail_case_file, 'w')
        if pass_case_file is not None:
            f_pass = open(pass_case_file, 'w')
        
        #set mode children into eval(): dropout, batchnorm,..
        self.model.eval()
        self.print_log('Evaluate epoch: {}'.format(epoch))
        for ln in loader_name:
            loss_value = []
            predictions = []
            step = 0
            process = tqdm(self.data_loader[ln])
            #_ is batch_idx
            for _, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = torch.tensor( data.float().to(self.device))
                    label = torch.tensor( label.long().to(self.device))
                    output = self.model(data)
                    
                    loss = self.loss(output, label)
                    predictions.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

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
            full_predictions = np.concatenate(predictions)
            full_loss = np.mean(loss_value)
            accuracy = self.__calculate_metric(full_predictions)
            if accuracy > self.best_acc["value"]:
                self.best_acc["value"] = accuracy
                self.best_acc["epoch"] = epoch

            
            if self.arg.phase == 'train':
                print('loss: {} epoch: {}'.format(full_loss, epoch))
                print('acc: {} epoch: {}'.format(accuracy, epoch))
            
            #scores is the highest value of predictions in each row:
            scores = torch.max(full_predictions,1)
            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, scores))
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        cfg_train_xview, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def train(self):
        losses=[]
        for epoch in self.num_of_epoch:
            for _, (data, label, _) in enumerate(tqdm(loader_train)):
                data = torch.tensor(data.float().to(self.device), requires_grad=False)
                label = torch.tensor(label.long().to(self.device), requires_grad=False)
                #forward
                output_ = self.model(data)
                loss_ = self.loss(output_, label)

                #backward
                self.optimizer.zero_grad()
                loss_.backward()
                self.optimizer.step()

                #asfvasfvavavdvvasv
                #asfvasfvavavdvvasv
                #asfvasfvavavdvvasv
                #cho nay tinh loss, acc va luu/visualize
                #asfvasfvavavdvvasv
                #asfvasfvavavdvvasv
                
            
                if (len(losses)%100!=0): continue
            plt.plot(losses)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig("output_train/losses{}.png".format(epoch))
        torch.save(self.model.state_dict(), "output_train/weight/model.pt")


if __name__ == "__main__":
    
    
