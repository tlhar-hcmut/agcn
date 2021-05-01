from torchsummary import summary
from src.main.config import cfg_train
from xcommon import xfile
import torch

import sys
import pytz
from datetime import datetime



output_architecture = cfg_train.output_train+"/"+ str(sys.argv[1])
xfile.mkdir(output_architecture)

class BaseTrainer:
    def __init__(self):
        self.model = None
        self.loss = None

    def summary_to_file(self, title=None, **kargs):
        if title is None: title=self.model.name
        with open(output_architecture + "/architecture.txt", 'a') as f:
            sys.stdout = f
            print("\n\n--------------------\n", datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')),": ", title, "\n--------------------\n")

            summary(model=self.model, depth=15, col_width=10, col_names=["input_size","kernel_size", "output_size", "num_params"],**kargs)
        
    def load_to_device(self):

        self.model.to(self.device)

        self.loss.to(self.device)

    def __calculate_metric(self, full_predictions: torch.tensor, loader_name):
        pass
    def __draw_confusion_matrix(self, epoch, full_predictions, loader_name):
        pass
    def evaluate(self, epoch, save_score, loader_name, fail_case_file, pass_case_file):
        self.model.eval()
        #do some thing here
        self.model.train()
    def train(self):
        pass
