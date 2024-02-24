from pathlib import Path
from typing import Dict, Optional

import torch
from clearml import Dataset as ClearmlDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import DataConfig
from src.dataset import VOCHumanBodyPart, VOCSegmentationBase
from src.transform import get_train_transforms, get_valid_transforms


class HumanBodyDataModule(LightningDataModule):  # noqa: WPS214
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self._train_transforms = get_train_transforms(*cfg.img_size)
        self._valid_transforms = get_valid_transforms(*cfg.img_size)

        # Prevent hyperparameters from being stored in checkpoints.
        self.save_hyperparameters(logger=False)

        self.data_path: Optional[Path] = None
        self.initialized = False

        self.data_train: Optional[VOCSegmentationBase] = None
        self.data_val: Optional[VOCSegmentationBase] = None
        self.data_test: Optional[VOCSegmentationBase] = None

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not self.initialized:
            self.prepare_data()
            self.setup('test')
        if self.data_test is None:
            raise ValueError('Test dataset needs to be initialized.')
        return self.data_test.class2idx

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {idx: cl for cl, idx in self.class_to_idx.items()}

    def prepare_data(self) -> None:
        self.data_path = Path(ClearmlDataset.get(dataset_name=self.cfg.dataset_name).get_local_copy())

    def setup(self, stage: str):
        if self.data_path is None:
            raise ValueError('Must call `prepare_data` before `setup`.')
        if stage == 'fit':
            all_data = VOCHumanBodyPart(
                str(self.data_path),
                image_list=self.data_path / 'pascal_person_part' / 'pascal_person_part_trainval_list' / 'train.txt',
                transform=self._train_transforms,
            )
            train_split = int(len(all_data) * self.cfg.data_split[0])
            val_split = len(all_data) - train_split
            self.data_train, self.data_val = torch.utils.data.random_split(  # noqa: WPS414
                all_data,
                [train_split, val_split],
            )
            self.data_val.transform = self._valid_transforms  # type: ignore
        elif stage == 'test':
            self.data_test = VOCHumanBodyPart(
                str(self.data_path),
                transform=self._valid_transforms,
                image_list=self.data_path
                / 'pascal_person_part'
                / 'pascal_person_part_trainval_list'
                / 'val.txt',  # FIXME merge train/val/test datasets and re-split them
                cache_size=0,
            )
        self.initialized = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )
