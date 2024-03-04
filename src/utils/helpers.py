import torch
import numpy as np
import pytorch_lightning as pl


def default(val, def_val):
    return def_val if val is None else val

def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epoch_warmup, max_epoch, min_lr = 1e-8):
        """
        Cosine learning rate scheduler with warmup.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer.
        epoch_warmup : int
            Number of epochs for warmup.
        max_epoch : int
            Maximum number of epochs the model is trained for.
        min_lr : float, optional
            Minimum learning rate. The default is 1e-9.
        """
        self.warmup = epoch_warmup
        self.max_num_iters = max_epoch
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch ==0 :
            return [self.min_lr]
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class SizeDatamodule(pl.callbacks.Callback):       
    def on_train_start(self, trainer: "pl.Trainer", 
                             pl_module:  "pl.LightningModule") -> None:
        """At each epoch start, log the size of the train set."""
        if pl_module.logger:
            pl_module.logger.log_metrics(
                {"train-size": print(len(trainer.train_dataloader.dataset)),
                 "val-size": len(trainer.val_dataloaders[0].dataset)},
                                         step=0)

    def on_test_start(self, trainer: "pl.Trainer", 
                      pl_module: "pl.LightningModule") -> None:
        """At each test start, log the size of the test set."""
        if pl_module.logger:
            pl_module.logger.log_metrics({"test-size":len(trainer.test_dataloaders[0].dataset)},
                                         step=0)
