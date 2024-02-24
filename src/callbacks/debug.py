from typing import Dict, Tuple

import albumentations as albu
import numpy as np
import torch.cuda
from albumentations.pytorch import ToTensorV2
from lightning import Callback, Trainer
from numpy.typing import NDArray
from torchvision.utils import make_grid

from src.dataset import VOCSegmentationBase
from src.lightning_module import SegmentationLightningModule


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: SegmentationLightningModule):  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs == 0:
            images, masks = next(iter(trainer.train_dataloader))

            base_images = []
            masked_images = []
            for img, mask in zip(images, masks):
                base_img = (
                    img.mul(255)
                    .add_(0.5)
                    .clamp_(0, 255)
                    .to(  # noqa: WPS221
                        pl_module.device,
                        torch.uint8,
                    )
                )
                base_images.append(base_img)
                rgb_mask = _tensor_mask_to_rgb(mask, VOCSegmentationBase.idx2color)
                masked_images.append(blend_mask_with_img_tensor(base_img, rgb_mask))

            image_grid = make_grid(base_images, normalize=False)
            trainer.logger.experiment.add_image(
                'Batch preview: images',
                img_tensor=image_grid,
                global_step=trainer.global_step,
            )

            masked_grid = make_grid(masked_images, normalize=True)
            trainer.logger.experiment.add_image(
                'Batch preview: masked',
                img_tensor=masked_grid,
                global_step=trainer.global_step,
            )


def _tensor_mask_to_rgb(
    mask: torch.Tensor,
    idx2color: Dict[int, Tuple[int, int, int]],
) -> NDArray[np.uint8]:
    arr = mask.permute(1, 2, 0).cpu().numpy()
    rgb_mask = np.zeros_like(arr)[:, :, :3]
    for idx, color in idx2color.items():
        rgb_mask[arr[:, :, idx] == 1] = color
    return rgb_mask


def blend_mask_with_img_tensor(
    img: torch.Tensor,
    mask: NDArray[np.uint8],
    alpha: float = 0.3,
    beta: float = 0.3,
) -> torch.Tensor:
    transform = albu.Compose([ToTensorV2()])
    mask_tensor = transform(image=mask)['image'].to(img.device)
    return alpha * img + beta * mask_tensor
