import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from datamodule import DataModule
from models import SimpleCNNModel

data_module = DataModule(batch_size=32)

# download dataset (CIFAR10)
data_module.prepare_data()

# make train-val-test data
data_module.setup()

# define model
model = SimpleCNNModel(input_shape=(3, 32, 32), num_class=data_module.num_class)

# configure wandb
wandb.login()
wandb_logger = WandbLogger(name='SimpleModel-32-0.001', project='advanced_deeplearning')

# configure callbacks (early-stopping)
early_stop_callback = EarlyStopping(monitor='val_loss')

# configure trainer
# https://pytorch-lightning.readthedocs.io/en/1.6.2/accelerators/gpu.html#select-gpu-devices (-1 : all available GPUs)
trainer = pl.Trainer(
    max_epochs=50,
    gpus='cpu',
    logger=wandb_logger,
    callbacks=[early_stop_callback]
)

# train model
trainer.fit(model, data_module)

# evaluate model
trainer.test(dataloaders=data_module.test_dataloader())

# close wandb
wandb.finish()