import logging
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from src.main.model import TKNet
from src.main.config import cfg_ds_v1
from src.main.config import cfg_train
from src.main.feeder.ntu import NtuFeeder
from src.main.graph import NtuGraph
from src.main.util import plot_confusion_matrix, setup_logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from xcommon import xfile

from src.main.app.base_train import BaseTrainer


output_train=cfg_train.output_train

xfile.mkdir(output_train)
xfile.mkdir(output_train+"/predictions")
xfile.mkdir(output_train+"/model")
xfile.mkdir(output_train+"/confusion_matrix")


class TrainXView(BaseTrainer):
    def __init__(self, pretrained_path=None):
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_of_epoch =200

        self.model = TKNet(stream=[1], num_class=12, cls_graph=NtuGraph)
        if (pretrained_path!=None):
            self.model.load_state_dict(torch.load(pretrained_path))

        _feeder_train = NtuFeeder(
            path_data=cfg_ds_v1.path_data_preprocess+"/train_xview_joint.npy",
            path_label=cfg_ds_v1.path_data_preprocess+"/train_xview_label.pkl",
        )
        _loader_train = DataLoader(
            dataset=_feeder_train,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
        _feeder_test = NtuFeeder(
            path_data=cfg_ds_v1.path_data_preprocess+"/val_xview_joint.npy",
            path_label=cfg_ds_v1.path_data_preprocess+"/val_xview_label.pkl",
        )
        _loader_test = DataLoader(
            dataset=_feeder_test,
            batch_size=1,
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
            "val_confusion": setup_logger(name="val_confusion_logger",
                                          log_file=output_train+"/confusion_val.log",
                                          level=logging.DEBUG),
            "train_confusion": setup_logger(name="train_confusion_logger",
                                            log_file=output_train+"/confusion_train.log",
                                            level=logging.DEBUG),

        }

        self.load_to_device()
        self.summary_to_file(self.model.name, model=self.model, input_size=self.model.input_size)



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

        logger.info('set: {} - epoch: {}\n'.format(loader_name, epoch) + str(df_confusion))
        plot_confusion_matrix(
            df_confusion, file_name=output_train+"/confusion_matrix/cf_mat_{}_{}.png".format(loader_name, epoch), title="confution matrix "+loader_name)

    def evaluate(self, epoch, save_score=False, loader_name=['val'], fail_case_file=None, pass_case_file=None):
        is_improved =False
        scl_loss_train = -1
        scl_loss_val = -1


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

            ls_loss = []
            ls_output = []
            process = tqdm(self.loader_data[ln])
            
            for _, (ts_data_batch, ts_label_batch, index) in enumerate(process):
                with torch.no_grad():
                    ts_data_batch = ts_data_batch.float().to(self.device)
                    ts_label_batch = ts_label_batch.long().to(self.device)
                    ts_output_batch = self.model(ts_data_batch)

                    scl_loss_batch = self.loss(ts_output_batch, ts_label_batch)
                    ls_output.append(ts_output_batch.data)
                    ls_loss.append(scl_loss_batch.data.item())

                    # get maximum in axes 1 (row): return max_values, max_indices
                    _, max_indices = torch.max(ts_output_batch.data, 1)

                if fail_case_file is not None or pass_case_file is not None:
                    ls_output_batch_scl = list(max_indices.cpu().numpy())
                    ls_label_batch = list(ts_label_batch.data.cpu().numpy())
                    for i, x in enumerate(ls_output_batch_scl):
                        if x == ls_label_batch[i] and pass_case_file is not None:
                            f_pass.write(
                                str(index[i]) + ',' + str(x) + ',' + str(ls_label_batch[i]) + '\n')
                        if x != ls_label_batch[i] and fail_case_file is not None:
                            f_fail.write(
                                str(index[i]) + ',' + str(x) + ',' + str(ls_label_batch[i]) + '\n')

            # concat output of each batch into single matrix
            ts_output = torch.cat(ls_output, dim=0)
            scl_loss = np.mean(ls_loss)

            if(ln=="train"):
                scl_loss_train=scl_loss
            else:
                scl_loss_val=scl_loss

            # log this every epoch
            scl_accuracy = self.__calculate_metric(
                full_predictions=ts_output, loader_name=ln)
            logger.info('loss: {} epoch: {}'.format(scl_loss, epoch))
            logger.info('acc: {} epoch: {}'.format(scl_accuracy, epoch))

            # draw confusion
            self.__draw_confusion_matrix( epoch=epoch, full_predictions=ts_output, loader_name=ln)
            
            # do this with only better performance
            if scl_accuracy > self.best_acc[ln]["value"]:
                self.best_acc[ln]["value"] = scl_accuracy
                self.best_acc[ln]["epoch"] = epoch

                # print vector predictions
                # predicted_labels = torch.max(ts_output, 1)
                # score_dict = dict(
                #     zip(self.loader_data[ln].dataset.sample_name, predicted_labels))
                # if save_score:
                #     with open('{}/epoch{}_{}_predict_vector.pkl'.format(
                #             output_train+"/predictions", epoch, ln), 'wb') as f:
                #         pickle.dump(score_dict, f)

                if (ln=="val"):
                    is_improved=True

            # do this at last epoch
            if (epoch == self.num_of_epoch):
                logger.info("The best accuracy: {}".format(
                    self.best_acc[ln]))

        return is_improved, scl_loss_train, scl_loss_val

    def train(self):
        ls_loss_train=[]
        ls_loss_val=[]
        for epoch in range(1, self.num_of_epoch+1):
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

            # evaluate every epoch
            is_store_model, scl_loss_train, scl_loss_val = self.evaluate(
                epoch,
                save_score=True,
                loader_name=[ "train", "val"],
                fail_case_file=output_train+"/result_fail.txt",
                pass_case_file=output_train+"/result_pass.txt"
            )

            ls_loss_train.append(scl_loss_train)
            ls_loss_val.append(scl_loss_val)
            
            # draw loss chart every 5-epoch
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(ls_loss_train, label="train")
            plt.plot(ls_loss_val, label="val")
            plt.legend(loc='best')

            plt.savefig(output_train+"/loss.png".format(epoch))

            if (is_store_model):
                torch.save(self.model.state_dict(),output_train+"/model/model_{}.pt".format(epoch))


if __name__ == "__main__":
    trainxview = TrainXView()
    trainxview.train()
