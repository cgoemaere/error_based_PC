from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import math

class BP(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        w_lr: float,
        w_decay: float = 0.0,
        output_loss = "mse",
        nm_batches=None,
        nm_epochs=None,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Store all layers and register them properly as parameters
        self.layers = torch.nn.ModuleList(architecture)

        self.errors = None  # Needs to be initialized with an input x

        self.w_lr = w_lr
        self.w_decay = w_decay

        if output_loss == "mse":
            mse = torch.nn.MSELoss(reduction="sum")
            self.output_loss = lambda y_pred, y: 0.5 * mse(y_pred, y)
        elif output_loss == "ce":
            self.output_loss = torch.nn.CrossEntropyLoss(reduction="sum")

        self.nm_batches = nm_batches
        self.nm_epochs = nm_epochs

    def y_pred(self, x: torch.Tensor):
        s_i = x
        for layer_i in self.layers:
            s_i = layer_i(s_i)
        return s_i

    def class_loss(self, y_pred: torch.Tensor, y: torch.Tensor):
        # For error optimization: reduction = "sum"
        # For weight optimization: reduction = "mean"
        # (but we just manually divide by batch_size in training_step)
        return self.output_loss(y_pred, y)

    def configure_optimizers(self):
        base_lr = self.w_lr
        peak_lr = 1.1 * base_lr
        end_lr = 0.1 * base_lr

        total_steps = self.nm_batches * self.nm_epochs
        warmup_steps = int(0.1 * total_steps)

        optimizer = torch.optim.Adam(self.layers.parameters(), lr=1.0, weight_decay=self.w_decay)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from base_lr to peak_lr
                return base_lr + (peak_lr - base_lr) * (current_step / warmup_steps)
            else:
                # Cosine decay from peak_lr to end_lr
                progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = end_lr + (peak_lr - end_lr) * cosine_decay
                return decayed

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda) # acts as multiplier of base lr set to 1.0
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }
        return torch.optim.Adam(self.layers.parameters(), lr=self.w_lr, weight_decay=self.w_decay)


    def on_fit_start(self):
        # Store batch_size for easy access
        self.batch_size = self.trainer.datamodule.batch_size

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx):
        y_pred = self.y_pred(x=batch["img"])

        E_final = self.class_loss(y_pred, y=batch["y"])

        self.log("E_local", E_final, prog_bar=True)

        # For weight optimization, we must average E over the batch.
        return E_final / self.batch_size  # = loss function for Lightning to minimize wrt params

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        
        # Log the dataset-specific metrics
        node_dict = {"y": self.y_pred(x=batch["img"])}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="val_"), prog_bar=True
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx):
        # Log the dataset-specific metrics
        node_dict = {"y": self.y_pred(x=batch["img"])}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="test_"), prog_bar=True
        )

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx):
        y_pred = self.y_pred(x=batch["img"])
        print("Loss =", self.class_loss(y_pred, y=batch["y"]).item())
        return {"y": y_pred}



class BPSkipConnection(BP):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        w_lr: float,
        w_decay: float = 0.0,
        output_loss = "mse",
        nm_batches=None,
        nm_epochs=None,

    ):
        super().__init__(architecture, w_lr, w_decay, output_loss, nm_batches, nm_epochs)

    def y_pred(self, x: torch.Tensor):
        s_i = (x, 0.0)  # activity, identity for skip connection
        for layer_i in self.layers:
            s_i = layer_i(s_i)  # layers take care of writing s_i[1] and adding it to s_i[0] 
        return s_i[0]
    
