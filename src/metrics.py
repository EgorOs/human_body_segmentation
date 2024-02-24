from torchmetrics import MetricCollection
from torchmetrics.classification import Dice


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'dice': Dice(**kwargs),
        },
    )
