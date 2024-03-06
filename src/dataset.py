from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import imageio
import jpeg4py as jpeg
import numpy as np
import torch
from jpeg4py import JPEGRuntimeError
from kornia import image_to_tensor
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from src.logger import LOGGER


class SegmentationBaseDataset(Dataset):
    idx2class: Dict[int, str] = {
        0: 'background',
        1: 'some_class',
    }
    idx2color: Dict[int, Tuple[int, int, int]] = {
        0: (0, 0, 0),
        1: (128, 0, 0),
    }

    def __init__(
        self,
        root: str | Path,
        size: Tuple[int, int],
    ):
        self.root = Path(root)
        self.size = size
        self.images: List[Path] = []
        self.targets: List[Path] = []

    @property
    def color2idx(self) -> Dict[Tuple[int, int, int], int]:
        inverted = {col: idx for idx, col in self.idx2color.items()}
        if len(inverted) != len(self.idx2color):
            raise ValueError('Duplicated colors were found')
        return inverted

    @property
    def class2idx(self) -> Dict[str, int]:
        inverted = {cl: idx for idx, cl in self.idx2class.items()}
        if len(inverted) != len(self.idx2class):
            raise ValueError('Duplicated class names were found')
        return inverted

    def validate_data(self) -> None:
        if not self.images or not self.targets:
            raise ValueError('Missing data')
        if len(self.images) > len(self.targets):
            raise ValueError(f'Data mismatch {len(self.images)=}, {len(self.targets)=}.')  # noqa: WPS237, WPS221

    def get_subset(self, indexes: List[int]) -> 'SegmentationBaseDataset':
        subset = deepcopy(self)
        subset.images = [self.images[idx] for idx in indexes]
        subset.targets = [self.targets[idx] for idx in indexes]
        return subset

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        try:
            img = jpeg.JPEG(self.images[idx]).decode()
        except JPEGRuntimeError:
            LOGGER.warning('Could not decode image %s, retrying with "imageio"', self.images[idx])
            img = imageio.v3.imread(self.images[idx])

        target = imageio.v3.imread(self.targets[idx])
        masks = rgb_to_one_hot(target, list(self.color2idx.keys()))

        img = image_to_tensor(img, keepdim=False).to(torch.float32) / 255  # noqa: WPS432
        masks = image_to_tensor(masks, keepdim=False)

        return img.squeeze(), masks.squeeze()


class VOCSegmentationBase(SegmentationBaseDataset):
    idx2class: Dict[int, str] = {
        0: 'background',
        1: 'head',
        2: 'upper_body',
        3: 'lower_body',
        4: 'upper_arm',
        5: 'lower_arm',
        6: 'lower_leg',
        # FIXME: NEED CLASS FOR FACE AND HAIR, right?
    }
    idx2color: Dict[int, Tuple[int, int, int]] = {
        0: (0, 0, 0),
        1: (128, 0, 0),
        2: (0, 128, 0),
        3: (128, 0, 128),
        4: (128, 128, 0),
        5: (0, 0, 128),
        6: (0, 128, 128),
    }

    def __init__(
        self,
        root: str | Path,
        image_list: str | Path,
        size: Tuple[int, int],
    ):
        super().__init__(root=root, size=size)

        self.root = Path(root)
        self.size = size

        img_tgt = _get_img_tgt_pairs(Path(image_list))
        self.images = [
            fpath
            for fpath in sorted(
                self.images_dir.glob('*'),
            )
            if fpath.name in {pair[0] for pair in img_tgt}
        ]
        self.targets = [
            fpath
            for fpath in sorted(
                self.targets_dir.glob('*'),
            )
            if fpath.name in {pair[1] for pair in img_tgt}
        ]

        self.validate_data()

    @property
    def images_dir(self) -> Path:
        return self.root / 'VOCdevkit' / 'VOC2010' / 'JPEGImages'

    @property
    def targets_dir(self) -> Path:
        return self.root / 'VOCdevkit' / 'VOC2010' / 'SegmentationClass'


class VOCHumanBodyPart(VOCSegmentationBase):
    @property
    def targets_dir(self) -> Path:
        return self.root / 'pascal_person_part' / 'pascal_person_part_gt'


class VITONHDSegmentation(SegmentationBaseDataset):
    # Based on
    # https://github.com/shadow2496/VITON-HD/blob/4261cd45949c93c355ca6b158bf988c49d2e4343/datasets.py#L156C9-L156C15
    idx2class = {
        0: 'background',
        1: 'hair_1',
        2: 'hair_2',
        3: 'noise_3',
        4: 'face_4',
        5: 'upper_5',
        6: 'upper_6',
        7: 'upper_7',
        8: 'socks',
        9: 'bottom_9',
        10: 'neck',
        11: 'noise_11',
        12: 'bottom_12',
        13: 'face_13',
        14: 'left_arm',
        15: 'right_arm',
        16: 'left_leg',
        17: 'right_leg',
        18: 'left_shoe',
        19: 'right_shoe',
    }

    idx2color = {
        0: (0, 0, 0),
        1: (128, 0, 0),
        2: (254, 0, 0),
        3: (0, 85, 0),
        4: (169, 0, 51),
        5: (254, 85, 0),
        6: (0, 0, 85),
        7: (0, 119, 220),
        8: (85, 85, 0),
        9: (0, 85, 85),
        10: (85, 51, 0),
        11: (52, 86, 128),
        12: (0, 128, 0),
        13: (0, 0, 254),
        14: (51, 169, 220),
        15: (0, 254, 254),
        16: (85, 254, 169),
        17: (169, 254, 85),
        18: (254, 254, 0),
        19: (254, 169, 0),
    }

    def __init__(
        self,
        root: str | Path,
        size: Tuple[int, int],
        subset: Literal['train', 'test'],
    ):
        super().__init__(root=root, size=size)

        self.root = Path(root)
        self.size = size
        self.subset = subset

        self.images = sorted(self.images_dir.glob('*'))
        self.targets = sorted(self.targets_dir.glob('*'))

        self.validate_data()

    @property
    def images_dir(self) -> Path:
        return self.root / self.subset / 'image'

    @property
    def targets_dir(self) -> Path:
        return self.root / self.subset / 'image-parse-v3'


def rgb_to_one_hot(
    mask_rgb: NDArray[np.uint8],
    pixel_values: Sequence[Tuple[int, int, int]],
) -> NDArray[np.uint8]:
    boolean_masks = [(mask_rgb == color).all(axis=-1) for color in pixel_values]
    return np.stack(boolean_masks, axis=-1).astype(np.uint8)


def _make_idx2layer(
    mask: NDArray[np.uint8],
    color2idx: Dict[Tuple[int, int, int], int],
) -> Dict[int, NDArray[np.uint8]]:
    height, width = mask.shape[:2]
    idx2layer = {}
    for color, idx in color2idx.items():
        layer = np.zeros((height, width, 1))
        layer[np.all(mask == color, axis=2)] = 1
        idx2layer[idx] = layer
    return idx2layer


def _get_img_tgt_pairs(image_list: str | Path) -> List[List[str]]:
    with open(image_list, 'r') as ann_file:
        return [
            [Path(fpath.strip()).name for fpath in record.split(' ')] for record in ann_file.readlines()  # noqa: WPS221
        ]
