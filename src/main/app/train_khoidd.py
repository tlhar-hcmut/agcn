from pytorch_lightning import Trainer, callbacks, loggers
from src.main.config import cfg_ds_v1, cfg_train
from src.main.feeder import NtuFeeder
from src.main.model import KhoiddNet
from torch.utils.data import DataLoader

if __name__ == "__main__":
    trainer: Trainer = Trainer(
        # general config
        max_epochs=200,
        auto_lr_find=True,
        logger=loggers.CSVLogger(save_dir="./log"),
        # gpu config
        gpus=-1,  # -1: train on all gpus
        precision=32,  # use amp
        # callback
        checkpoint_callback=callbacks.ModelCheckpoint(
            dirpath=cfg_train.output_train + "/model", monitor="val_loss", mode="min",
        ),
        # only use when debug
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        check_val_every_n_epoch=1,  # epoch per val
        val_check_interval=1.0,  # val per epoch
    )
    trainer.fit(
        model=KhoiddNet(),
        train_dataloader=DataLoader(
            dataset=NtuFeeder(
                path_data=cfg_ds_v1.path_data_preprocess + "/train_xview_joint.npy",
                path_label=cfg_ds_v1.path_data_preprocess + "/train_xview_label.pkl",
                random_speed=True,
            ),
            batch_size=156,
        ),
        val_dataloaders=DataLoader(
            dataset=NtuFeeder(
                path_data=cfg_ds_v1.path_data_preprocess + "/val_xview_joint.npy",
                path_label=cfg_ds_v1.path_data_preprocess + "/val_xview_label.pkl",
            ),
            batch_size=156,
        ),
    )
