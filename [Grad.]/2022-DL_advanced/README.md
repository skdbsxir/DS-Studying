# 1st toy project using PyTorch-Lightning 

## Code structure
  - datamodule.py : Defines DataLoader objects.
    - creates train_dataloader, val_dataloader, and test_dataloader
  - models.py : Defines simple CNN model.
    - 3 Conv2d layers
    - 2 MaxPool2d layers
    - 3 Dense layers (includes output layer)
    - Used **torchmetric's Accuracy module** for calculate classification score.
  - main.py : Train & Test model on CIFAR10 dataset.
    - Includes WandB logger & EarlyStopping.

<br>

## Configuration
  - 1 GPU
  - Optimizer : Adam with lr = 1e-4
  - Batch size : 32
  - Early Stopping (by monitoring validation loss)

<br>

## Some Notations & Reports
- Code has executed in Cheetah container.
- WandB report : **[wandb link](https://wandb.ai/happysky12/advanced_deeplearning)**  
- Train & Val chart : **[Chart view](https://wandb.ai/happysky12/advanced_deeplearning/reports/Train-Val-22-09-29-21-17-23---VmlldzoyNzE2Njg0)**
