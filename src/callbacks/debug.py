from typing import Dict, List, Optional, Tuple

import albumentations as albu
import numpy as np
import torch.cuda
from albumentations.pytorch import ToTensorV2
from lightning import Callback, Trainer
from numpy.typing import NDArray
from torchvision.utils import draw_segmentation_masks, make_grid

from src.dataset import VOCSegmentationBase
from src.lightning_module import SegmentationLightningModule, prob_to_mask
from src.transform import DENORMALIZE


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: SegmentationLightningModule):  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs == 0:
            batch = next(iter(trainer.train_dataloader))

            log_batch(batch, 'Batch', trainer, pl_module)

            transformed_batch = pl_module.on_after_batch_transfer(batch, dataloader_idx=0)
            log_batch(transformed_batch, 'Transformed batch', trainer, pl_module, denormalize=True)


class VisualizePredictions(Callback):
    def __init__(  # noqa: WPS234
        self,
        threshold: float,
        colors: Optional[List[Tuple[int, ...]]] = None,
        idx_to_drop_bg: Optional[int] = None,
    ):
        super().__init__()
        self.threshold = threshold
        self.colors = colors
        self.idx_to_drop_bg = idx_to_drop_bg

        if idx_to_drop_bg is not None and self.colors is not None:
            self.colors.pop(idx_to_drop_bg)

        self.done_logging: bool = False

    def on_validation_batch_end(  # noqa: WPS211, WPS210
        self,
        trainer: Trainer,
        pl_module: SegmentationLightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.done_logging:
            return

        images = batch[0]
        masks = prob_to_mask(outputs, threshold=self.threshold)
        if self.idx_to_drop_bg is not None:
            idx = self.idx_to_drop_bg
            masks = torch.cat([masks[:, :idx], masks[:, idx + 1 :]], dim=1)  # noqa: WPS221

        visualizations = []
        for img, mask in zip(images, masks):
            visualizations.append(
                draw_segmentation_masks(
                    img.add(0.5)
                    .mul(255)
                    .clamp(0, 255)
                    .to(
                        torch.uint8,
                    )
                    .cpu(),
                    mask.to(torch.bool).cpu(),
                    alpha=0.75,
                    colors=self.colors,
                ),
            )
        image_grid = make_grid(visualizations, normalize=False)
        trainer.logger.experiment.add_image(
            'Predictions',
            img_tensor=image_grid,
            global_step=trainer.global_step,
        )
        self.done_logging = True

    def on_validation_end(self, trainer: Trainer, pl_module: SegmentationLightningModule) -> None:
        self.done_logging = False


def log_batch(  # noqa: WPS210
    batch: Tuple[torch.Tensor, torch.Tensor],
    prefix: str,
    trainer: Trainer,
    pl_module: SegmentationLightningModule,
    denormalize: bool = False,
) -> None:
    images, masks = batch

    base_images = []
    masked_images = []
    for img, mask in zip(images, masks):
        if denormalize:
            img = DENORMALIZE(img)[0]
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
        f'{prefix}: images',
        img_tensor=image_grid,
        global_step=trainer.global_step,
    )
    masked_grid = make_grid(masked_images, normalize=True)
    trainer.logger.experiment.add_image(
        f'{prefix}: masked',
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
