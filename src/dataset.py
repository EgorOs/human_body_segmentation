from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import imageio
import jpeg4py as jpeg
import numpy as np
import torch
from kornia import image_to_tensor
from kornia.geometry import resize as korn_resize
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


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
        cache_size: int = 0,
    ):
        self.root = Path(root)
        self.size = size
        self.cache_size = cache_size
        self._data_cache: Dict[int, Any] = {}
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

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        cached_items = self._data_cache.get(idx)

        if cached_items:
            img, masks = cached_items
        else:
            img = jpeg.JPEG(self.images[idx]).decode()
            target = imageio.v3.imread(self.targets[idx])
            masks = rgb_to_one_hot(target, list(self.color2idx.keys()))
            if len(self._data_cache) < self.cache_size:
                self._data_cache[idx] = (img, masks)

        img = image_to_tensor(img, keepdim=False).to(torch.float32) / 255  # noqa: WPS432
        masks = image_to_tensor(masks, keepdim=False)

        img = korn_resize(img, self.size)  # FIXME, run resizing on GPU
        masks = korn_resize(masks, self.size, interpolation='nearest')
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
        cache_size: int = 0,
    ):
        super().__init__(root=root, size=size)

        self.root = Path(root)
        self.size = size
        self.cache_size = cache_size
        self._data_cache: Dict[int, Any] = {}

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
    @property
    def images_dir(self) -> Path:
        # FIXME: parametrize 'train | test'
        return self.root / 'train' / 'image'

    @property
    def targets_dir(self) -> Path:
        return self.root / 'train' / 'image-parse-v3'


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
