from torchmetrics import MetricCollection
from torchmetrics.classification import Dice


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'dice_metric': Dice(**kwargs),
        },
    )
