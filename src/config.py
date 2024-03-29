from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.deserialization import load_object


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # Disallow unexpected arguments.


class DataConfig(_BaseValidatedConfig):
    dataset_name: str = 'pascal_parts_dataset'
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    data_split: Tuple[float, ...] = (0.7, 0.2, 0.1)
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2

    @model_validator(mode='after')  # type: ignore
    def splits_add_up_to_one(self) -> 'DataConfig':
        epsilon = 1e-5
        total = sum(self.data_split)
        if abs(total - 1) > epsilon:
            raise ValueError(f'Splits should add up to 1, got {total}.')
        return self


class ObjToInit(_BaseValidatedConfig):
    target: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    def instantiate(self, **kwargs: Any) -> Any:
        merged_kwargs = self.kwargs | kwargs
        return load_object(self.target)(**merged_kwargs)


class ModuleConfig(_BaseValidatedConfig):
    segm_kwargs: Dict[str, Any] = Field(
        default={
            'arch': 'FPN',
            'encoder_name': 'efficientnet-b0',
        },
    )
    optimizer: ObjToInit = Field(default=ObjToInit(target='torch.optim.AdamW', kwargs={'lr': 1e-3}))
    scheduler: ObjToInit = Field(
        default=ObjToInit(
            target='src.schedulers.get_cosine_schedule_with_warmup',
            kwargs={
                'num_warmup_steps': 30,
                'num_cycles': 1.8,
            },
        ),
    )


class TrainerConfig(_BaseValidatedConfig):
    min_epochs: int = 7  # prevents early stopping
    max_epochs: int = 20

    # perform a validation loop every N training epochs
    check_val_every_n_epoch: int = 3

    log_every_n_steps: int = 50

    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None

    # set True to ensure deterministic results
    # makes training slower but gives more reproducibility than just setting seeds
    deterministic: bool = False

    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None

    detect_anomaly: bool = False


class ExperimentConfig(_BaseValidatedConfig):
    project_name: str = 'human_body_segmentation'
    experiment_name: str = 'segmentation_baseline'
    track_in_clearml: bool = True
    trainer_config: TrainerConfig = Field(default=TrainerConfig())
    data_config: DataConfig = Field(default=DataConfig())
    module_config: ModuleConfig = Field(default=ModuleConfig())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]):
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.model_dump(), out_file, default_flow_style=False, sort_keys=False)
