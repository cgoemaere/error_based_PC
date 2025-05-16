from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import math

class PCS(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        s_lr: float,
        w_lr: float,
        w_decay: float = 0.0,
        s_momentum: Optional[float] = None,
        output_loss = "mse",
        nm_batches=None,
        nm_epochs=None,

    ):
        super().__init__()

        self.save_hyperparameters()

        # Store all layers and register them properly as parameters
        self.layers = torch.nn.ModuleList(architecture)

        self.iters = iters
        self.s_lr = s_lr
        self.w_lr = w_lr
        self.w_decay = w_decay
        self.s_momentum = s_momentum

        if output_loss == "mse":
            mse = torch.nn.MSELoss(reduction="sum")
            self.output_loss = lambda y_pred, y: 0.5 * mse(y_pred, y)
        elif output_loss == "ce":
            # self.output_loss = torch.nn.CrossEntropyLoss(reduction="sum")
            self.output_loss = lambda y_pred, y: - (y * F.log_softmax(y_pred, dim=-1)).sum()

        self.nm_batches = nm_batches
        self.nm_epochs = nm_epochs

    
    def class_loss(self, y_pred: torch.Tensor, y: torch.Tensor):
        # For error optimization: reduction = "sum"
        # For weight optimization: reduction = "mean"
        # (but we just manually divide by batch_size in training_step)
        return self.output_loss(y_pred, y)

    def configure_optimizers(self):
        bass_lr = self.w_lr
        peak_lr = 1.1 * bass_lr
        end_lr = 0.1 * bass_lr

        total_steps = self.nm_batches * self.nm_epochs
        warmup_steps = int(0.1 * total_steps)

        optimizer = torch.optim.Adam(self.layers.parameters(), lr=1.0, weight_decay=self.w_decay, decoupled_weight_decay=True)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from bass_lr to peak_lr
                return bass_lr + (peak_lr - bass_lr) * (current_step / warmup_steps)
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
        # return torch.optim.Adam(self.layers.parameters(), lr=self.w_lr, weight_decay=self.w_decay)

    def y_pred(self, x: torch.Tensor):
        for layer_i in self.layers:
            x = layer_i(x)
        return x
    
    def E_states_only(self, x: torch.Tensor, y: torch.Tensor, states: list[torch.Tensor]):
        """
        Calculates the energy using only the states, which need to be given as inputs.
        No errors are used here.
        """
        losses = [lambda y1, y2: 0.5*torch.nn.MSELoss(reduction="sum")(y1, y2)] * len(states) + [self.class_loss]
        states = [x] + states + [y]

        return sum(
            loss(layer(s_i), s_ip1)
            for s_i, s_ip1, layer, loss in zip(states[:-1], states[1:], self.layers, losses)
        )

    def minimize_state_energy(self, x: torch.Tensor, y: torch.Tensor, iters: int, s_lr: float):
        """Classical PC energy minimization using states"""

        # Deactivate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(False)

        # Initialize states using a feedforward pass
        def ff_init(s):
            return [(s := layer(s).detach().requires_grad_(True)) for layer in self.layers[:-1]]

        states = ff_init(x)

        # Minimize energy via the states
        state_optim = torch.optim.SGD(states, lr=s_lr, momentum=self.s_momentum)
        for _ in range(iters):
            state_optim.zero_grad()
            E = self.E_states_only(x, y, states)
            E.backward()
            state_optim.step()

        # Re-activate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(True)

        # No need to store in self.states, just return for later use in callbacks
        return states

    def on_fit_start(self):
        # Store batch_size for easy access
        self.batch_size = self.trainer.datamodule.batch_size


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx):
        states = self.minimize_state_energy(
            x=batch["img"],
            y=batch["y"],
            iters=self.iters,
            s_lr=self.s_lr,
        )

        E_final = self.E_states_only(
            x=batch["img"],
            y=batch["y"],
            states=states,
        )

        self.log("E_local", E_final, prog_bar=True)

        # For weight optimization, we must average E over the batch.
        return E_final / self.batch_size  # = loss function for Lightning to minimize wrt params

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        y_pred = self.y_pred(x=batch["img"])

        # Log the dataset-specific metrics
        node_dict = {"y": y_pred}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="val_"), prog_bar=True
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx):
        y_pred = self.y_pred(x=batch["img"])

        # Log the dataset-specific metrics
        node_dict = {"y": y_pred}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="test_"), prog_bar=True
        )

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx):
        y_pred = self.y_pred(x=batch["img"])

        print("Loss =", self.class_loss(y_pred, y=batch["y"]).item())
        return {"y": y_pred}



class PCSSkipConnection(PCS):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        s_lr: float,
        w_lr: float,
        w_decay: float = 0.0,
        s_momentum: Optional[float] = None,
        output_loss = "mse",
        nm_batches=None,
        nm_epochs=None,

    ):
        super().__init__(architecture, iters, s_lr, w_lr, w_decay, s_momentum, output_loss, nm_batches, nm_epochs)

    def y_pred(self, x: torch.Tensor):
        s_i = (x, 0.0)  # activity, identity for skip connection
        for layer_i in self.layers:
            s_i = layer_i(s_i)  # layers take care of writing s_i[1] and adding it to s_i[0]
        return s_i[0]
    
    def E_states_only(self, x: torch.Tensor, y: torch.Tensor, states: list[torch.Tensor]):
        """
        Calculates the energy using only the states, which need to be given as inputs.
        No errors are used here.
        """
        losses = [lambda y1, y2: 0.5*torch.nn.MSELoss(reduction="sum")(y1, y2)] * len(states) + [self.class_loss]
        states = [x] + states + [y]

        errors =[]
        identity = 0.0
        for s_i, s_ip1, layer, loss in zip(states[:-1], states[1:], self.layers, losses):
            x_ = (s_i, identity)
            x_ = layer(x_) 
            errors.append(loss(x_[0], s_ip1))
            identity = x_[1]  # identity is the second element of the tuple

        return sum(errors)
