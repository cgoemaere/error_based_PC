import lightning
import wandb
from custom_callbacks import ErrorConvergenceCallback
from datamodules import CIFAR10, CIFAR100, TinyImageNet
from get_arch import get_architecture, get_cnn_architecture
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from pc_e import PCE
from torch import nn
import torch


def train_model(logger, run_config):
    # Make sure to always generate the *exact* same datasets & batches
    lightning.seed_everything(run_config["seed"], workers=True)

    # 1: load dataset as Lightning DataModule
    batch_size = run_config["batch_size"]
    if run_config["dataset"] == "CIFAR10":
        datamodule = CIFAR10(batch_size, is_test=run_config["is_test"])
    elif run_config["dataset"] == "CIFAR100":
        datamodule = CIFAR100(batch_size, is_test=run_config["is_test"])
    elif run_config["dataset"] == "tiny-imagenet":
        datamodule = TinyImageNet(batch_size, is_test=run_config["is_test"])
    print("Training on", datamodule.dataset_name)

    # 2: Set up Lightning trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[ErrorConvergenceCallback(), lr_monitor],
        max_epochs=run_config["nm_epochs"],
        inference_mode=False,  # inference_mode would interfere with the state backward pass
        limit_predict_batches=1,  # enable 1-batch prediction
        num_sanity_val_steps=0,
    )

    # 3: Get architecture that belongs to this dataset
    architecture = get_cnn_architecture(run_config["model"], datamodule.dataset_name,run_config["act_fn"])

    # 4: Initiate model and train it
    datamodule.setup("fit")
    pc = PCE(
        architecture,
        iters=run_config["iters"],
        e_lr=run_config["e_lr"],
        w_lr=run_config["w_lr"],
        w_decay=run_config["w_decay"],
        output_loss=run_config["output_loss"],
        nm_batches=len(datamodule.train_dataloader()),
        nm_epochs=run_config["nm_epochs"],
    )
    trainer.fit(pc, datamodule=datamodule)

    # 5: Test results
    trainer.test(pc, datamodule=datamodule)

    # 6: Release all CUDA memory that you can
    pc = None
    trainer = None
    lightning.pytorch.utilities.memory.garbage_collection_cuda()



if __name__ == "__main__":
    config = {
        "seed": 42,
        "batch_size": 256,
        "nm_epochs": 25,
        "iters": 5,
        "e_lr": 0.001,
        "w_lr": 0.000662772765622318,
        "w_decay": 0.0003639117865323884,
        "output_loss": "ce",
        "model": "VGG5",
        "act_fn": "gelu",
        "dataset": "CIFAR10",
        "is_test": True,
    }

    wandb.init(project="ErrorPC", entity="oliviers-gaspard")
    logger = WandbLogger(project="ErrorPC", entity="oliviers-gaspard", mode="online")
    logger_config = logger.experiment.config

    # overwrite config with logger config if it exists
    for key, value in logger_config.items(): 
        config[key] = value
    logger.experiment.config.update(config) # update wandb config with the full config

    train_model(logger, config)
    wandb.finish()