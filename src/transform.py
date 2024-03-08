from typing import Tuple

import kornia.augmentation as korn
import kornia.constants
import torch
import torch.nn as nn

NORMALIZE = korn.Normalize(
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor(
        [0.229, 0.224, 0.225],
    ),
)


DENORMALIZE = korn.Denormalize(
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor(
        [0.229, 0.224, 0.225],
    ),
)


class TrainTransform(nn.Module):
    def __init__(self, img_width: int, img_height: int):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height

        self.appearance = korn.AugmentationSequential(
            korn.ColorJitter(0.2, 0.3, 0.2, 0.2, p=0.3),
            korn.RandomClahe(p=0.2),
            korn.RandomBoxBlur((3, 3), p=0.2),
            korn.RandomHue(hue=(-0.3, 0.3), p=0.1),
            NORMALIZE,
        )

        self.geometry = self.create_geometry_augmentations(kornia.constants.Resample.BILINEAR)

    def create_geometry_augmentations(self, resample: kornia.constants.Resample) -> korn.AugmentationSequential:
        return korn.AugmentationSequential(
            korn.RandomHorizontalFlip(p=0.5),
            korn.RandomPerspective(distortion_scale=0.5, resample=resample, p=0.5),
            korn.RandomRotation(degrees=(-10, 10), resample=resample, p=0.5),
            # TODO: not quite sure about positive effect of `RandomErasing` transformation, need to validate
            korn.RandomErasing((0.1, 0.5), (0.3, 1 / 0.3), p=0.5),
            korn.RandomResizedCrop(
                size=(self.img_width, self.img_height),
                resample=resample,
                scale=(0.8, 1.2),
                p=1,
            ),
            korn.PadTo(size=(self.img_width, self.img_height), pad_value=0),
            data_keys=['image', 'mask'],
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.geometry(img, mask)
        img = self.appearance(img)
        return img, mask


class ValidTransform(nn.Module):
    def __init__(self, img_width: int, img_height: int):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height

        self.appearance = korn.AugmentationSequential(
            NORMALIZE,
        )

        self.geometry = self.create_geometry_augmentations(kornia.constants.Resample.BILINEAR)

    def create_geometry_augmentations(self, resample: kornia.constants.Resample) -> korn.AugmentationSequential:
        return korn.AugmentationSequential(
            korn.Resize(
                size=(self.img_width, self.img_height),
                resample=resample,
                p=1,
            ),
            data_keys=['image', 'mask'],
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.geometry(img, mask)
        img = self.appearance(img)
        return img, mask
