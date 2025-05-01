import torch
from torch.optim import Adam

class AdamW(Adam):
    '''
        This is the pytorch equivalent of adamw in optax.
        the weight decay is applied after the adam normalisation step.
        This helps with the PC models because the gradients can have small values which causes decay to dominate.
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, amsgrad=amsgrad)
        self.actual_weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)

        # Apply weight decay AFTER normalization step
        for group in self.param_groups:
            wd = self.actual_weight_decay
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue
                # Apply decay directly to parameters after update
                param.data.mul_(1 - lr * wd)

        return loss
