# Human body part segmentation

<a href="https://www.pytorchlightning.ai/index.html"><img alt="PytorchLightning" src="https://img.shields.io/badge/PytorchLightning-7930e3?logo=lightning&style=flat"></a>
<a href="https://github.com/qubvel/segmentation_models.pytorch"><img alt="PytorchSegmentationModels" src="https://img.shields.io/badge/SegmentationModels-dfe6e9?logo=PyTorch&style=flat"></a>
<a href="https://clear.ml/docs/latest/"><img alt="Config: Hydra" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

# Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org)
   to install Poetry:
   ```bash
   # Unix/MacOs installation
   curl -sSL https://install.python-poetry.org | python3 -
   ```
1. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Setup workspace:
   ```bash
   make setup_ws
   ```
1. Setup ClearML:
   ```bash
   clearml-init
   ```
1. Migrate dataset to your ClearML workspace:
   ```bash
   make migrate_dataset
   ```

# Train

```bash
make run_training
```
