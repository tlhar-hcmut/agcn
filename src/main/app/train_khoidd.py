from pytorch_lightning import Trainer, callbacks
from src.main.config import cfg_ds_v1, cfg_train
from src.main.feeder import NtuFeeder
from src.main.model import KhoiddNet
from torch.utils.data import DataLoader

if __name__ == "__main__":
    trainer: Trainer = Trainer(
        gpus=-1,  # -1: train on all gpus
        use_amp=True,
        max_epochs=200,
        # callback
        checkpoint_callback=callbacks.ModelCheckpoint(
            filepath=cfg_train.output_train + "/model",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        # only use when debug
        fast_dev_run=False,
        show_progress_bar=True,
        train_percent_check=1.0,  # percent of train data
        val_percent_check=1.0,  # percent of val data
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
            batch_size=cfg_train.batch_size,
        ),
        val_dataloaders=DataLoader(
            dataset=NtuFeeder(
                path_data=cfg_ds_v1.path_data_preprocess + "/val_xview_joint.npy",
                path_label=cfg_ds_v1.path_data_preprocess + "/val_xview_label.pkl",
            ),
            batch_size=cfg_train.batch_size,
        ),
    )
