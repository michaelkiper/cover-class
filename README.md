# Package

## Installation

## Example Usage
### Train Time
```
from cover_class.train import setup_training_from_config

dataloader, test_set, train_set = setup_training_from_config(
    '/my/path/config.yaml',
    batch_size,
    shuffle = True,
)
```