from torchsummary import summary
from src.main.config import cfg_train
from xcommon import xfile
import torch
import datetime

import sys



output_architecture = cfg_train.output_train+"/"+ str(sys.argv[1])
xfile.mkdir(output_architecture)

class BaseTrainer:
    def __init__(self):
        self.model = None
        self.loss = None

    def summary_to_file(self, title=None, **kargs):
        with open(cfg_train.output_train + "/architecture.txt", 'a') as f:
            sys.stdout = f
            print("\n\n--------------------\n", datetime.datetime.now(),": ", title, "\n--------------------\n")

            summary(**kargs)
        
    def load_to_device(self):

        self.model.to(self.device)

        self.loss.to(self.device)

    def __calculate_metric(self, full_predictions: torch.tensor, loader_name):
        pass
    def __draw_confusion_matrix(self, epoch, full_predictions, loader_name):
        pass
    def evaluate(self, epoch, save_score, loader_name, fail_case_file, pass_case_file):
        pass
    def train(self):
        pass
