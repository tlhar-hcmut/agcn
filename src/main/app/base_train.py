from torchsummary import summary
from src.main.config import cfg_train
from xcommon import xfile
import torch
from src.main.config import cfg_ds_v1
from src.main.config import *
from src.main.feeder.ntu import NtuFeeder
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from src.main.util import plot_confusion_matrix, setup_logger
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
from torch import nn







import sys
import pytz
from datetime import datetime
import os



class BaseTrainer:
    def __init__(self, models, cfgs):
        
        self.cfgs   = cfgs
        for cfg in self.cfgs:
            os.makedirs(cfg.output_train, exist_ok=True)
            os.makedirs(cfg.output_train+"/predictions", exist_ok=True)
            os.makedirs(cfg.output_train+"/model", exist_ok=True)
            os.makedirs(cfg.output_train+"/confusion_matrix", exist_ok=True)  

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = models
        self.num_model = len(self.models)

        for i in range(self.num_model):
            if (self.cfgs[i].pretrained_path != None):
                self.models[i].load_state_dict(torch.load(self.cfgs[i].pretrained_path))

        self.lossfuncs = [get_loss(x.loss) for x in self.cfgs]
        if len(self.lossfuncs)==1: self.lossfuncs = self.lossfuncs*self.num_model

        self.best_accs = [{"train": {"value": 0, "epoch": 0},
                        "val": {"value": 0, "epoch": 0}} for _ in range(self.num_model)]

        self.loggers = [TrainLogger(
                        val     = setup_logger( name="val_logger"+x.name,
                                                log_file=x.output_train+"/eval_val.log",
                                                level=logging.DEBUG),
                        train   = setup_logger( name="train_logger"+x.name,
                                                log_file=x.output_train+"/eval_train.log",
                                                level=logging.DEBUG),
                        val_confusion   = setup_logger( name="val_confusion_logger"+x.name,
                                                        log_file=x.output_train+"/confusion_val.log",
                                                        level=logging.DEBUG),
                        train_confusion=setup_logger(name="train_confusion_logger"+x.name,
                                                        log_file=x.output_train+"/confusion_train.log",
                                                        level=logging.DEBUG),
                    ) for x in self.cfgs]
       
        self.loader_data= load_data(cfg_ds_v1, CfgTrain.batch_size)

        self.optimizers = [load_optim(cfg.optim)(model.parameters()) for (model,cfg) in zip(self.models,self.cfgs)]
    
        self.load_to_device()
        self.summary_to_file()

    def summary_to_file(self):
        for i in range(self.num_model):
            name=self.cfgs[i].name
            desc = self.cfgs[i].desc
            with open(self.cfgs[i].output_train + "/architecture.txt", 'a') as f:
                sys.stdout = f
                print('{:-<100}'.format(""))
                print(datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')))
                print('model name:\t '+name)
                print('description:\t '+desc)
                print('{:-<100}'.format(""))

                summary(model=self.models[i], depth=15, col_width=20, col_names=["input_size","kernel_size", "output_size", "num_params"], input_data=torch.empty((1,*self.cfgs[i].input_size)))
        
    def load_to_device(self):
        
        [x.to(self.device) for x in self.models]
        [x.to(self.device) for x in self.lossfuncs]

    def __calculate_metric(self, full_predictions: torch.tensor, loader_name='val'):
        true_labels = torch.tensor(self.loader_data[loader_name].dataset.label).to(self.device)
        predict_labels = torch.argmax(full_predictions, 1).to(self.device)
        hit_cases = true_labels == predict_labels

        #accuracy = true_prediction/ total_predictions
        acc = sum(hit_cases) * 1.0 / len(hit_cases)
        return acc.item()

    def __draw_confusion_matrix(self, epoch=-1, full_predictions=None, loader_name='val', logger=None, output_train=None):


        true_labels = torch.tensor(self.loader_data[loader_name].dataset.label).to("cpu")
        predict_labels = torch.argmax(full_predictions, 1).to("cpu")

        df_true_labels = pd.Series(true_labels, name='Actual')
        df_predict_labels = pd.Series(predict_labels, name='Predicted')
        df_confusion = pd.crosstab(df_true_labels, df_predict_labels, margins=True)

        logger.info('set: {} - epoch: {}\n'.format(loader_name, epoch) + str(df_confusion))
        plot_confusion_matrix(df_confusion, file_name=output_train+"/confusion_matrix/cf_mat_{}_{}.png".format(loader_name, epoch), title="confution matrix "+loader_name)
    
    def evaluate(self, epoch, save_score=False, loader_name=['val']):
        [x.eval() for x in self.models]

        ls_is_improved = [False]*self.num_model

        ls_scl_loss_train = []
        ls_scl_loss_val = []
        
        for i in range(self.num_model):
            for ln in loader_name:
                logger = getattr(self.loggers[i], ln)
                if (epoch == 1):    logger.info("----------------- start -----------------")

                ls_loss = []
                ls_output = []
                process = tqdm(self.loader_data[ln])
                
                for _, (ts_data_batch, ts_label_batch, index) in enumerate(process):
                    with torch.no_grad():
                        ts_data_batch = ts_data_batch.float().to(self.device)
                        ts_label_batch = ts_label_batch.long().to(self.device)
                        ts_output_batch = self.models[i](ts_data_batch)

                        scl_loss_batch = self.lossfuncs[i](ts_output_batch, ts_label_batch)
                        ls_output.append(ts_output_batch.data)
                        ls_loss.append(scl_loss_batch.data.item())

                # concat output of each batch into single matrix
                ts_output = torch.cat(ls_output, dim=0)
                scl_loss = np.mean(ls_loss)

                if(ln=="train"):
                    ls_scl_loss_train.append(scl_loss)
                else:
                    ls_scl_loss_val.append(scl_loss)

                # draw confusion
                logger = getattr(self.loggers[i],ln+"_confusion")
                self.__draw_confusion_matrix( epoch=epoch, full_predictions=ts_output, loader_name=ln, logger=logger, output_train=self.cfgs[i].output_train)
                
                # do this with only better performance
                logger = getattr(self.loggers[i], ln)
                scl_accuracy = self.__calculate_metric(full_predictions=ts_output, loader_name=ln)
                if scl_accuracy > self.best_accs[i][ln]["value"]:
                    self.best_accs[i][ln]["value"] = scl_accuracy
                    self.best_accs[i][ln]["epoch"] = epoch
                    logger.info('epoch: {:<5}loss: {:<10}acc: {:<10} {:-<10}BEST'.format(epoch, round(scl_loss,5), round(scl_accuracy,5),""))
                else:
                    logger.info('epoch: {:<5}loss: {:<10}acc: {:<10} '.format(epoch, round(scl_loss,5), round(scl_accuracy,5)))

                    if (ln=="val"):
                        ls_is_improved[i]=True

        [x.eval() for x in self.models]
        return ls_is_improved, ls_scl_loss_train, ls_scl_loss_val

    def train(self):
        ls_ls_loss_train    = [[] for _ in range(self.num_model)]
        ls_ls_loss_val      = [[] for _ in range(self.num_model)]
        for epoch in range(1, CfgTrain.num_of_epoch+1):
            for _, (data, label, _) in enumerate(tqdm(self.loader_data["train"])):
                data = data.float().to(self.device)
                data.requires_grad = False
                label = label.long().to(self.device)
                label.requires_grad = False

                # forward
                ls_output_batch = [model(data) for model in self.models]
                ls_loss_batch = [lossfunc(output, label) for (lossfunc, output) in zip(self.lossfuncs,ls_output_batch)]

                # backward
                [x.zero_grad()  for x in self.optimizers]
                [x.backward()  for x in ls_loss_batch]
                [x.step()       for x in self.optimizers]

            # evaluate every epoch
            ls_is_store_model, ls_scl_loss_train, ls_scl_loss_val = self.evaluate(
                epoch,
                save_score=True,
                loader_name=[ "train", "val"],
            )

            [ls_ls_loss_train[i].append(x)  for i,x in enumerate(ls_scl_loss_train)]
            [ls_ls_loss_val[i].append(x)     for i,x in enumerate(ls_scl_loss_val)]
            
            for i in range(self.num_model):
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(ls_ls_loss_train[i], label="train")
                plt.plot(ls_ls_loss_val[i], label="val")
                plt.legend(loc='best')

                plt.savefig(self.cfgs[i].output_train+"/loss{}.png".format(epoch))
                plt.close()
                if epoch >1:
                    os.remove(self.cfgs[i].output_train+"/loss{}.png".format(epoch-1))

                if (ls_is_store_model[i]):
                    torch.save(self.models[i].state_dict(),self.cfgs[i].output_train+"/model/model_{}.pt".format(epoch))

class TrainLogger:
    def __init__(self, val, train, val_confusion, train_confusion):
        self.val    = val  
        self.train  = train 
        self.val_confusion      = val_confusion
        self.train_confusion    = train_confusion
    

def load_data(cfg, batch_size):
        
        _feeder_train = NtuFeeder(
            path_data=cfg.path_data_preprocess+"/train_xview_joint.npy",
            path_label=cfg.path_data_preprocess+"/train_xview_label.pkl",
            random_speed=True
        )
        _loader_train = DataLoader(
            dataset=_feeder_train,
            batch_size=CfgTrain.batch_size,
            shuffle=False,
            num_workers=2,
        )
        _feeder_test = NtuFeeder(
            path_data=cfg.path_data_preprocess+"/val_xview_joint.npy",
            path_label=cfg.path_data_preprocess+"/val_xview_label.pkl",
            # random_speed=True
        )
        _loader_test = DataLoader(
            dataset=_feeder_test,
            batch_size=CfgTrain.batch_size,
            shuffle=False,
            num_workers=2,
        )
        return {"train": _loader_train, "val": _loader_test}

def load_optim(optims):
    if optims=="adam":
        return optim.Adam
    
def get_loss(loss):
    if loss=="crossentropy":
        return nn.CrossEntropyLoss()
