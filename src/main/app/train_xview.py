import logging
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from src.main.model.net import Net
from src.main.config import cfg_ds_v1
from src.main.feeder.ntu import NtuFeeder
from src.main.graph import NtuGraph
from src.main.model.agcn import UnitAGCN
from src.main.util import plot_confusion_matrix, setup_logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from xcommon import xfile

# output_train = "/content/gdrive/Shareddrives/Thesis/result_bert/two_stream"
output_train = "output"
xfile.mkdir(output_train)
xfile.mkdir(output_train+"/predictions")
xfile.mkdir(output_train+"/loss")
xfile.mkdir(output_train+"/confusion_matrix")


class TrainXView:
    def __init__(self, pretrained_path=None):
            
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_of_epoch =100

        self.model = Net(self.device, num_class=12, cls_graph=NtuGraph)
        if (pretrained_path!=None):
            self.model.load_state_dict(torch.load(pretrained_path))

        _feeder_train = NtuFeeder(
            path_data=cfg_ds_v1.path_data_preprocess+"/val_xview_joint.npy",
            path_label=cfg_ds_v1.path_data_preprocess+"/val_xview_label.pkl",
        )
        _loader_train = DataLoader(
            dataset=_feeder_train,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )
        _feeder_test = NtuFeeder(
            path_data=cfg_ds_v1.path_data_preprocess+"/train_xview_joint.npy",
            path_label=cfg_ds_v1.path_data_preprocess+"/train_xview_label.pkl",
        )
        _loader_test = DataLoader(
            dataset=_feeder_test,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )
        self.loader_data: Dict = {"train": _loader_train, "val": _loader_test}

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        self.loss = nn.CrossEntropyLoss()

        self.best_acc = {"train": {"value": 0, "epoch": 0},
                         "val": {"value": 0, "epoch": 0}}

        self.logger = {
            "val": setup_logger(name="val_logger",
                                log_file=output_train+"/eval_val.log",
                                level=logging.DEBUG),
            "train": setup_logger(name="train_logger",
                                  log_file=output_train+"/eval_train.log",
                                  level=logging.DEBUG),
            "val_confusion": setup_logger(name="train_confusion_logger",
                                          log_file=output_train+"/confusion_val.log",
                                          level=logging.DEBUG),
            "train_confusion": setup_logger(name="train_confusion_logger",
                                            log_file=output_train+"/confusion_train.log",
                                            level=logging.DEBUG),

        }

    def __load_to_device(self):

        self.model.to(self.device)

        self.loss.to(self.device)

    def __calculate_metric(self, full_predictions: torch.tensor, loader_name='val'):
        true_labels = torch.tensor(self.loader_data[loader_name].dataset.label).to(self.device)
        predict_labels = torch.argmax(full_predictions, 1).to(self.device)
        hit_cases = true_labels == predict_labels

        #accuracy = true_prediction/ total_predictions
        acc = sum(hit_cases) * 1.0 / len(hit_cases)
        return acc

    def __draw_confusion_matrix(self, epoch=-1, full_predictions=None, loader_name='val'):

        logger = self.logger[loader_name+"_confusion"]

        true_labels = torch.tensor(
            self.loader_data[loader_name].dataset.label).to("cpu")
        predict_labels = torch.argmax(full_predictions, 1).to("cpu")

        df_true_labels = pd.Series(true_labels, name='Actual')
        df_predict_labels = pd.Series(predict_labels, name='Predicted')
        df_confusion = pd.crosstab(
            df_true_labels, df_predict_labels, margins=True)

        logger.info('epoch: {}\n'.format(epoch) + str(df_confusion))
        plot_confusion_matrix(
            df_confusion, file_name=output_train+"/confusion_matrix/cf_mat_{}_{}.png".format(loader_name, epoch), title="confution matrix "+loader_name)

    def evaluate(self, epoch, save_score=False, loader_name=['val'], fail_case_file=None, pass_case_file=None):
        is_improved =False

        if fail_case_file is not None:
            f_fail = open(fail_case_file, 'w')
        if pass_case_file is not None:
            f_pass = open(pass_case_file, 'w')

        # set mode children into eval(): dropout, batchnorm,..
        self.model.eval()
        for ln in loader_name:
        
            logger = self.logger[ln]
            if (epoch == 1):
                logger.info("----------------- start -----------------")
            logger.info('Evaluate epoch: {}'.format(epoch))

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

                    # get maximum in axes 1 (row): return max_values, max_indices
                    _, max_indices = torch.max(output.data, 1)
                    step += 1

                if fail_case_file is not None or pass_case_file is not None:
                    predictions = list(max_indices.cpu().numpy())
                    lables = list(label.data.cpu().numpy())
                    for i, x in enumerate(predictions):
                        if x == lables[i] and pass_case_file is not None:
                            f_pass.write(
                                str(index[i]) + ',' + str(x) + ',' + str(lables[i]) + '\n')
                        if x != lables[i] and fail_case_file is not None:
                            f_fail.write(
                                str(index[i]) + ',' + str(x) + ',' + str(lables[i]) + '\n')

            # concat output of each batch into single matrix
            full_outputs = torch.cat(output_batch_list, dim=0)
            full_loss = np.mean(loss_value_list)

            # log this every epoch
            accuracy = self.__calculate_metric(
                full_predictions=full_outputs, loader_name=ln)
            logger.info('loss: {} epoch: {}'.format(full_loss, epoch))
            logger.info('acc: {} epoch: {}'.format(accuracy, epoch))

            # do this with only better performance
            if accuracy > self.best_acc[ln]["value"]:
                self.best_acc[ln]["value"] = accuracy
                self.best_acc[ln]["epoch"] = epoch

                # print vector predictions
                predicted_labels = torch.max(full_outputs, 1)
                score_dict = dict(
                    zip(self.loader_data[ln].dataset.sample_name, predicted_labels))
                if save_score:
                    with open('{}/epoch{}_{}_predict_vector.pkl'.format(
                            output_train+"/predictions", epoch, ln), 'wb') as f:
                        pickle.dump(score_dict, f)

                # draw confusion
                self.__draw_confusion_matrix(
                    epoch=epoch, full_predictions=full_outputs, loader_name=ln)

                if (ln=="train"):
                    is_improved=True

            # do this at last epoch
            if (epoch == self.num_of_epoch):
                logger.info("The best accuracy: {}".format(
                    self.best_acc[ln]))

        return is_improved

    def train(self):
        losses = []
        self.__load_to_device()
        for epoch in range(1, self.num_of_epoch+1):
            losses_epoch = []
            for _, (data, label, _) in enumerate(tqdm(self.loader_data["train"])):
                data = data.float().to(self.device)
                data.requires_grad = False
                label = label.long().to(self.device)
                label.requires_grad = False

                # forward
                output_batch = self.model(data)
                loss_batch = self.loss(output_batch, label)

                # backward
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()
                losses_epoch.append(loss_batch.item())

            # evaluate every epoch
            is_store_model = self.evaluate(
                epoch,
                save_score=True,
                loader_name=[ "train", "val"],
                fail_case_file=output_train+"/result_fail.txt",
                pass_case_file=output_train+"/result_pass.txt"
            )

            # draw loss chart every 5-epoch
            losses.append(sum(losses_epoch)/len(losses_epoch))
            if (epoch % 5 == 0 or epoch == self.num_of_epoch):
                plt.plot(losses)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.savefig(
                    output_train+"/loss/losses_{}.png".format(epoch))
                if (is_store_model):
                    torch.save(self.model.state_dict(),output_train+"/model_{}.pt".format(epoch))


if __name__ == "__main__":
    trainxview = TrainXView()
    trainxview.train()
