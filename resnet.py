import lightning
import wandb
from custom_callbacks import ErrorConvergenceCallback
from datamodules import CIFAR100, CIFAR10, TinyImageNet
from get_arch import get_resnet_architecture
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pc_e import PCESkipConnection
from torch import nn

def train_model(logger, run_config):
    # Make sure to always generate the *exact* same datasets & batches
    lightning.seed_everything(run_config["seed"], workers=True)

    # 1: load dataset as Lightning DataModule
    batch_size = run_config["batch_size"]
    datamodule = TinyImageNet(batch_size)
    print("Training on", datamodule.dataset_name)

    # 2: Set up Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[ErrorConvergenceCallback()],
        max_epochs=run_config["nm_epochs"],
        inference_mode=False,  # inference_mode would interfere with the state backward pass
        limit_predict_batches=1,  # enable 1-batch prediction
        num_sanity_val_steps=0,
    )

    # 3: Get architecture that belongs to this dataset
    architecture = get_resnet_architecture(run_config["model"], datamodule.dataset_name)

    # 4: Initiate model and train it
    pc = PCESkipConnection(
        architecture,
        iters=run_config["iters"],
        e_lr=run_config["e_lr"],
        w_lr=run_config["w_lr"],
        w_decay=run_config["w_decay"],
        output_loss=run_config["output_loss"],
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
        "nm_epochs": 50,
        "iters": 5,
        "e_lr": 0.001,
        "w_lr": 0.0001,
        "w_decay": 0.0,
        "output_loss": "mse",
        "model": "ResNet18",
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