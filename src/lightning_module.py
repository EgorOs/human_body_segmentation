from typing import Any, Dict, List, Tuple

import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from segmentation_models_pytorch.losses._functional import (  # noqa: WPS436
    focal_loss_with_logits,
    soft_dice_score,
)
from torch import Tensor
from torchmetrics import MeanMetric

from src.metrics import get_metrics
from src.schedulers import get_cosine_schedule_with_warmup
from src.transform import TrainTransform, ValidTransform


class SegmentationLightningModule(LightningModule):  # noqa: WPS214
    def __init__(self, class_to_idx: Dict[str, int], img_size: Tuple[int, int]):
        super().__init__()
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()
        self.train_transform: torch.nn.Module = TrainTransform(*img_size)
        self.valid_transform: torch.nn.Module = ValidTransform(*img_size)

        metrics = get_metrics(
            num_classes=None,
            multiclass=True,
            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix='valid_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.model = smp.create_model(
            arch='FPN',
            encoder_name='efficientnet-b0',
            classes=len(class_to_idx),
        )  # FIXME parametrize

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def on_after_batch_transfer(self, batch: List[Tensor], dataloader_idx: int) -> Tuple[Tensor, Tensor]:
        orig_dtype = batch[0].dtype
        images, targets = batch
        transform = self.train_transform if self.trainer.training else self.valid_transform

        # perform GPU/Batched data augmentation
        images, targets = transform(
            images.double(),
            targets.double(),
        )

        return (
            images.to(orig_dtype),
            targets.to(orig_dtype),
        )

    def training_step(self, batch: List[Tensor]) -> Dict[str, Tensor]:  # noqa: WPS210
        images, targets = batch
        logits = self(images)
        loss = self.calculate_loss(logits, targets, prefix='train')
        self._train_loss(loss)
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._train_loss.reset()

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        self._valid_loss(self.calculate_loss(logits, targets, prefix='valid'))

        self._valid_metrics(logits, targets.to(torch.int32))

    def on_validation_epoch_end(self) -> None:
        self.log('mean_valid_loss', self._valid_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._valid_loss.reset()

        self.log_dict(self._valid_metrics.compute(), prog_bar=True, on_epoch=True)
        self._valid_metrics.reset()

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        images, targets = batch
        logits = self(images)

        preds = torch.argmax(logits, dim=1)
        self._test_metrics(logits, targets.to(torch.int32))
        return preds

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), prog_bar=True, on_epoch=True)
        self._test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        # TODO: parametrize optimizer and lr scheduler.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)  # noqa: WPS432 will be parametrized
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,  # noqa: WPS432 will be parametrized
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.4,  # noqa: WPS432 will be parametrized
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def calculate_loss(self, logits: Tensor, targets: Tensor, prefix: str) -> Tensor:
        probs = torch.nn.functional.sigmoid(logits)
        losses = {
            f'{prefix}_dice_loss': soft_dice_score(probs, targets),
            f'{prefix}_focal_loss': focal_loss_with_logits(logits, targets),
        }
        losses[f'{prefix}_total_loss'] = sum(loss for loss in losses.values())
        self.log_dict(losses, prog_bar=True, on_epoch=True)
        return losses[f'{prefix}_total_loss']
