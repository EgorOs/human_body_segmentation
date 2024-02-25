import os

import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.callbacks.debug import VisualizeBatch, VisualizePredictions
from src.callbacks.experiment_tracking import ClearMLTracking
from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import HumanBodyDataModule
from src.dataset import VOCSegmentationBase
from src.lightning_module import SegmentationLightningModule


def train(cfg: ExperimentConfig):
    lightning.seed_everything(0)
    datamodule = HumanBodyDataModule(cfg=cfg.data_config)
    datamodule.prepare_data()

    callbacks = [
        VisualizeBatch(every_n_epochs=10),
        VisualizePredictions(
            threshold=0.5,
            idx_to_drop_bg=0,
            colors=list(VOCSegmentationBase.idx2color.values()),
        ),  # Fixme parametrize
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(save_top_k=3, monitor='valid_dice_metric', mode='max', every_n_epochs=1),
    ]
    if cfg.track_in_clearml:
        tracking_cb = ClearMLTracking(cfg, label_enumeration=datamodule.class_to_idx)
        callbacks += [
            tracking_cb,
        ]
    model = SegmentationLightningModule(class_to_idx=datamodule.class_to_idx, img_size=cfg.data_config.img_size)

    trainer = Trainer(**dict(cfg.trainer_config), callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
