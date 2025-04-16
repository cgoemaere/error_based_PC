import lightning
import wandb
from custom_callbacks import ErrorConvergenceCallback
from datamodules import EMNIST
from get_arch import get_architecture
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pc_e import PCE


# Define training function
def wandb_run_sweep():
    # Make sure to always generate the *exact* same datasets & batches
    lightning.seed_everything(42, workers=True)

    logger = WandbLogger(project="PredictiveCoding", entity="hopfield", mode="online")
    run_config = logger.experiment.config

    # 1: load dataset as Lightning DataModule
    batch_size = 64
    datamodule = EMNIST(batch_size)
    print("Training on", datamodule.dataset_name)

    # 2: Set up Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[ErrorConvergenceCallback()],
        max_epochs=2,
        inference_mode=False,  # inference_mode would interfere with the state backward pass
        limit_predict_batches=1,  # enable 1-batch prediction
    )

    # 3: Get architecture that belongs to this dataset
    architecture = get_architecture(dataset=datamodule.dataset_name)

    # 4: Initiate model and train it
    pc = PCE(
        architecture,
        iters=run_config["iters"],
        e_lr=run_config["e_lr"],
        w_lr=run_config["w_lr"],
    )
    trainer.fit(pc, datamodule=datamodule)

    # 5: Test results
    trainer.test(pc, datamodule=datamodule)

    # 6: Release all CUDA memory that you can
    pc = None
    trainer = None
    lightning.pytorch.utilities.memory.garbage_collection_cuda()


def main():
    wandb.login()

    # Define the search space
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "errors/max_grad"},
        "parameters": {
            "iters": {"values": [4, 8, 16, 32, 64, 128]},
            "e_lr": {"values": [float(f"{m}e{e}") for m in [1, 5] for e in range(-3, 0)]},
            "w_lr": {"value": 0.001},
        },
    }

    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="PredictiveCoding")
    wandb.agent(sweep_id, function=wandb_run_sweep)


if __name__ == "__main__":
    main()
