import os

import lightning

from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import HumanBodyDataModule
from src.dataset import VOCHumanBodyPart


def train(cfg: ExperimentConfig):
    lightning.seed_everything(0)
    datamodule = HumanBodyDataModule(cfg=cfg.data_config)
    datamodule.prepare_data()
    print(datamodule.data_path)
    train_set = VOCHumanBodyPart(
        datamodule.data_path,
        image_list=datamodule.data_path / 'pascal_person_part' / 'pascal_person_part_trainval_list' / 'train.txt',
        transforms=None,
    )
    batch = train_set[0]

    # callbacks = [
    #     VisualizeBatch(every_n_epochs=5),
    #     LearningRateMonitor(logging_interval='step'),
    #     ModelCheckpoint(save_top_k=3, monitor='valid_f1', mode='max', every_n_epochs=1),
    # ]
    # if cfg.track_in_clearml:
    #     tracking_cb = ClearMLTracking(cfg, label_enumeration=datamodule.class_to_idx)
    #     callbacks += [tracking_cb, LogConfusionMatrix(tracking_cb, datamodule.idx_to_class)]
    # model = ClassificationLightningModule(class_to_idx=datamodule.class_to_idx)
    #
    # trainer = Trainer(**dict(cfg.trainer_config), callbacks=callbacks, overfit_batches=60)
    # trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
