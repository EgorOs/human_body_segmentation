import random
from pathlib import Path
from typing import Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import DataConfig
from src.dataset import VITONHDSegmentation, VOCSegmentationBase
from src.logger import LOGGER
from src.preprocessing import resize_images


class HumanBodyDataModule(LightningDataModule):  # noqa: WPS214
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg

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
        orig_data_path = Path('/home/egor/Projects/ladi-vton-triton-serving/VITON-HD')  # FIXME implement download

        img_size = self.cfg.img_size
        data_path = orig_data_path / f'{img_size[0]}x{img_size[1]}'

        if data_path.is_dir():
            LOGGER.info('Found images of the desired size at %s', data_path)
        else:
            LOGGER.info("Couldn't find images of the desired size at %s, resizing...", data_path)
            for subset in ('train', 'test'):
                resize_images(orig_data_path / subset / 'image', data_path / subset / 'image', img_size)
                resize_images(
                    orig_data_path / subset / 'image-parse-v3',
                    data_path / subset / 'image-parse-v3',
                    img_size,
                )

        self.data_path = data_path

    def setup(self, stage: str):
        if self.data_path is None:
            raise ValueError('Must call `prepare_data` before `setup`.')
        if stage == 'fit':
            all_data = VITONHDSegmentation(
                str(self.data_path),
                size=self.cfg.img_size,
                subset='train',
                cache_size=0,
            )
            train_split = int(len(all_data) * self.cfg.data_split[0])

            all_indexes = list(range(len(all_data)))
            random.shuffle(all_indexes)

            self.data_train = all_data.get_subset(indexes=all_indexes[:train_split])
            self.data_val = all_data.get_subset(indexes=all_indexes[train_split:])

        elif stage == 'test':
            self.data_test = VITONHDSegmentation(
                str(self.data_path),
                size=self.cfg.img_size,
                subset='test',
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
            persistent_workers=True,  # For cache to work
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=True,  # For cache to work
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            persistent_workers=True,  # For cache to work
        )
