from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import math

from utils import AdamW

class PCE(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        e_lr: float,
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

        self.iters = iters
        self.e_lr = e_lr
        self.w_lr = w_lr
        self.w_decay = w_decay

        if output_loss == "mse":
            mse = torch.nn.MSELoss(reduction="sum")
            self.output_loss = lambda y_pred, y: 0.5 * mse(y_pred, y)
        elif output_loss == "ce":
            self.output_loss = torch.nn.CrossEntropyLoss(reduction="sum")

        self.nm_batches = nm_batches 
        self.nm_epochs = nm_epochs

        self.energy_scale = min([1.0, e_lr * iters]) # to avoid tiny errors from inference

    def y_pred(self, x: torch.Tensor):
        s_i = x
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)
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

    def E(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only the errors

        DANGER: don't use this E to train the params, or you'll be backpropping!
        """
        E_errors = 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in self.errors)

        return E_errors + self.class_loss(self.y_pred(x), y)

    def E_local(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only local interactions (no backprop!)
        Specifically, it infers the states from the errors and returns the states-based energy.

        By construction, the value is exactly equal to the energy using only errors,
        but its computational graph is different and enforces local weight updates.
        """
        E = 0.0
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)  # tracking the computational graph...
            s_i = (e_i + s_i_pred).detach()  # detach => no backprop!

            E += 0.5 * F.mse_loss(s_i_pred, s_i, reduction="sum")

        y_pred = self.layers[-1](s_i)
        return E + self.class_loss(y_pred, y)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is None:
            # Inference is easy: all errors are zero
            self.errors = [0.0] * (len(self.layers) - 1)

        else:  # Training is more difficult
            self.minimize_error_energy(x, y)

        # We don't need to return anything during training.
        # At inference, we can easily access the error values through self.errors

    def minimize_error_energy(self, x: torch.Tensor, y: torch.Tensor):
        """Novel PC energy minimization, using errors instead of states"""
        # Deactivate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(False)

        # Initialize self.errors to the right shape using a forward pass
        self.init_zero_errors(x)

        # Minimize energy via the errors
        error_optim = torch.optim.SGD(self.errors, lr=self.e_lr)
        for _ in range(self.iters):
            error_optim.zero_grad()
            E = self.E(x, y)
            E.backward()
            error_optim.step()

        # Log final energy
        self.log("E_errors", E, prog_bar=True)

        # Re-activate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def init_zero_errors(self, x: torch.Tensor):
        """Creates trainable errors via a feedforward pass"""
        self.errors = [
            torch.zeros_like(x := layer_i(x), requires_grad=True) for layer_i in self.layers[:-1]
        ]

    def on_fit_start(self):
        # Store batch_size for easy access
        self.batch_size = self.trainer.datamodule.batch_size

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"], y=batch["y"])

        # IMPORTANT: calculate the energy using the states!
        # (needed for local weight updates + good sanity check)
        E_final = self.E_local(x=batch["img"], y=batch["y"])

        self.log("E_local", E_final, prog_bar=True)

        # For weight optimization, we must average E over the batch.
        return E_final / (self.batch_size * self.energy_scale)  # = loss function for Lightning to minimize wrt params

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"])

        # Log the dataset-specific metrics
        node_dict = {"y": self.y_pred(x=batch["img"])}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="val_"), prog_bar=True
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"])

        # Log the dataset-specific metrics
        node_dict = {"y": self.y_pred(x=batch["img"])}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="test_"), prog_bar=True
        )

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"])

        print("Loss =", self.class_loss(self.y_pred(x=batch["img"]), y=batch["y"]).item())
        return {"y": self.y_pred(x=batch["img"])}


class PCESkipConnection(PCE):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        e_lr: float,
        w_lr: float,
        w_decay: float = 0.0,
        output_loss = "mse",
        nm_batches=None,
        nm_epochs=None,

    ):
        super().__init__(architecture, iters, e_lr, w_lr, w_decay, output_loss, nm_batches, nm_epochs)

    def y_pred(self, x: torch.Tensor):
        s_i = (x, 0.0)  # activity, identity for skip connection
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = layer_i(s_i)  # layers take care of writing s_i[1] and adding it to s_i[0]
            s_i = (s_i[0] + e_i, s_i[1]) 
        return s_i[0]
    

    def E_local(self, x, y):
        E = 0.0
        s_i = (x, 0.0)
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)  # tracking the computational graph...
            s_i = (e_i + s_i_pred[0]).detach()  # detach => no backprop!

            E += 0.5 * F.mse_loss(s_i_pred[0], s_i, reduction="sum")
            s_i = (s_i, s_i_pred[1]) 

        y_pred = self.layers[-1](s_i)[0]
        return E + self.class_loss(y_pred, y)
