from torchsummary import summary
from src.main.config import cfg_train
from xcommon import xfile
import torch

import sys
import pytz
from datetime import datetime



output_architecture = cfg_train.output_train
if len(list(sys.argv))>1:
    output_architecture+="/" + str(sys.argv[1])
xfile.mkdir(output_architecture)

class BaseTrainer:
    def __init__(self):
        self.models = []
        self.cfgs   = []
        self.losses = []
        self.num_model = len(self.models)
        self.loggers = []
        
    def summary_to_file(self, title=None, **kargs):
        if title is None: title=self.model.name
        with open(output_architecture + "/architecture.txt", 'a') as f:
            sys.stdout = f
            print("\n\n--------------------\n", datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')),": ", title, "\n--------------------\n")

            summary(model=self.model, depth=15, col_width=20, col_names=["input_size","kernel_size", "output_size", "num_params"],**kargs)
        
    def load_to_device(self):
        
        [x.to(self.device) for x in self.models]
        [x.to(self.device) for x in self.losses]

    def __calculate_metric(self, full_predictions: torch.tensor, loader_name):
        pass
    def __draw_confusion_matrix(self, epoch, full_predictions, loader_name):
        pass
    
    def evaluate(self, epoch, save_score=False, loader_name=['val'], fail_case_file=None, pass_case_file=None):
        [x.eval() for x in self.models]

        is_improved =[False]*self.num_model

        scl_loss_train = [-1]*self.num_model
        scl_loss_val = [-1]*self.num_model
        
        for ln in loader_name:
        
            logger = self.logger[ln]
            if (epoch == 1):
                logger.info("----------------- start -----------------")

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

            # draw confusion
            self.__draw_confusion_matrix( epoch=epoch, full_predictions=ts_output, loader_name=ln)
            
            # do this with only better performance
            if scl_accuracy > self.best_acc[ln]["value"]:
                self.best_acc[ln]["value"] = scl_accuracy
                self.best_acc[ln]["epoch"] = epoch
                logger.info('epoch: {:<5}loss: {:<10}acc: {:<10} {:-<10}BEST'.format(epoch, round(scl_loss,5), round(scl_accuracy,5),""))
            else:
                logger.info('epoch: {:<5}loss: {:<10}acc: {:<10} '.format(epoch, round(scl_loss,5), round(scl_accuracy,5)))


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

        self.model.train()
        return is_improved, scl_loss_train, scl_loss_val

    def train(self):
        pass
