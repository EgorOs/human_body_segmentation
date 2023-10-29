from os import PathLike
from pathlib import Path
from typing import Callable, List, Optional, Set

from torch.utils.data import Dataset

from src.constants import IMG_EXTENSIONS


class VOCSegmentationBase(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        image_list: str | PathLike,
        transforms: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transforms = transforms

        img_tgt = self._get_img_tgt_pairs(Path(image_list))
        self.images = [
            fpath for fpath in sorted(self.images_dir.glob('*')) if fpath.name in {pair[0] for pair in img_tgt}
        ]
        self.targets = [
            fpath for fpath in sorted(self.targets_dir.glob('*')) if fpath.name in {pair[1] for pair in img_tgt}
        ]

        if not self.images or not self.targets:
            raise ValueError('Missing data')
        if len(self.images) != len(self.targets):
            raise ValueError(f'Data mismatch {len(self.images)=}, {len(self.targets)=}.')

    @property
    def images_dir(self) -> Path:
        return self.root / 'VOCdevkit' / 'VOC2010' / 'JPEGImages'

    @property
    def targets_dir(self) -> Path:
        return self.root / 'VOCdevkit' / 'VOC2010' / 'SegmentationClass'

    @staticmethod
    def _get_img_tgt_pairs(image_list: str | PathLike) -> List[List[str]]:
        with open(image_list, 'r') as file:
            return [
                [Path(fpath.strip()).name for fpath in record.split(' ')] for record in file.readlines()  # noqa: WPS221
            ]

    @staticmethod
    def _load_images(folder: Path, extensions: Set[str] = IMG_EXTENSIONS) -> List[Path]:
        return [fpath for fpath in folder.glob('*') if fpath.suffix in extensions]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Path:
        return self.images[idx], self.targets[idx]


class VOCHumanBodyPart(VOCSegmentationBase):
    @property
    def targets_dir(self) -> Path:
        return self.root / 'pascal_person_part' / 'pascal_person_part_gt'
