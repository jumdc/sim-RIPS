import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD, Adam
from utils import default
import torchvision.models as models

from torch import einsum

class SimCLR_pl(pl.LightningModule):
    def __init__(self, cfg, model=None, feat_dim=512):
        super().__init__()
        self.cfg = cfg
        
        self.model = Encoder(cfg, model=model, mlp_dim=feat_dim)
        self.loss = symInfoNCE(cfg.batch_size, temperature=self.cfg)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]



class Encoder(nn.Module):
   def __init__(self, config, model=None, mlp_dim=512):
       super().__init__()
       embedding_size = config.embedding_size
       self.backbone = default(model, models.resnet18(pretrained=False, num_classes=config.embedding_size))
       mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
       print('Dim MLP input:',mlp_dim)
       self.backbone.fc = nn.Identity()

       # add mlp projection head
       self.projection = nn.Sequential(
           nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
           nn.BatchNorm1d(mlp_dim),
           nn.ReLU(),
           nn.Linear(in_features=mlp_dim, out_features=embedding_size),
           nn.BatchNorm1d(embedding_size),
       )

   def forward(self, x, return_embedding=False):
       embedding = self.backbone(x)
       if return_embedding:
           return embedding
       return self.projection(embedding)
   

class symInfoNCE(nn.Module):
    """symmetric infoNCE loss, aka same scheme than in ImageBind"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        logit_scale = torch.ones([]) * self.cfg.model.contrastive_loss.temperature
        self.gamma = cfg.model.contrastive_loss.gamma
        if cfg.model.contrastive_loss.learnable_scale and not cfg.model.contrastive_loss.cos:
            self.temperature = nn.Parameter(logit_scale)
        elif not cfg.model.contrastive_loss.cos : 
            self.temperature = logit_scale
        else :
            self.temperature_max = self.cfg.model.contrastive_loss.temperature_max
            self.temperature_min = self.cfg.model.contrastive_loss.temperature_min
            self.period = self.cfg.model.contrastive_loss.period

    def forward(self, x_1, x_2, current_epoch=0):
        if self.gamma != 0:
            x_1 = torch.nn.functional.normalize(x_1, dim=-1, p=2)
            x_2 = torch.nn.functional.normalize(x_2, dim=-1, p=2)
            noise_to_add = torch.normal(0, 1, size=x_1.shape, device=x_1.device)
            x_1 = x_1 + noise_to_add * self.gamma
            noise_to_add_2 = torch.normal(0, 1, size=x_2.shape, device=x_2.device)
            x_2 = x_2 + noise_to_add_2 * self.gamma
        x_1 = torch.nn.functional.normalize(x_1, dim=-1, p=2)
        x_2 = torch.nn.functional.normalize(x_2, dim=-1, p=2)
        labels = torch.arange(x_1.shape[0], device=x_1.device) # entries on the diagonal
        similarity = einsum('i d, j d -> i j', x_1, x_2) / self.temperature
        loss_1 = torch.nn.functional.cross_entropy(similarity, labels) 
        loss_2 = torch.nn.functional.cross_entropy(similarity.T, labels) 
        return (loss_1 + loss_2) / 2.0
    

def define_param_groups(model, weight_decay, optimizer_name):
   def exclude_from_wd_and_adaptation(name):
       if 'bn' in name:
           return True
       if optimizer_name == 'lars' and 'bias' in name:
           return True

   param_groups = [
       {
           'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
           'weight_decay': weight_decay,
           'layer_adaptation': True,
       },
       {
           'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
           'weight_decay': 0.,
           'layer_adaptation': False,
       },
   ]
   return param_groups