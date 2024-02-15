import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam
from utils.helpers import default, CosineWarmupScheduler
import torchvision.models as models
from torchmetrics import Accuracy, F1Score
from torch import einsum

class SimCLR_pl(pl.LightningModule):
    def __init__(self, cfg, model=None, feat_dim=512, stage="self-supervised"):
        super().__init__()
        self.cfg = cfg
        self.model = Encoder(self.cfg, model=model, mlp_dim=feat_dim)
        self.loss = symInfoNCE(self.cfg)
        self.stage = stage

    def forward(self, X):
        if self.stage == "self-supervised":
            output =  self.model(X)
        else :
            output = self.classifier(self.model(X))
        return output
    
    def make_classifier(self):
        self.model.projection = nn.Identity()
        self.classifier = nn.Linear(self.cfg.representation_size, 
                                    self.cfg.num_classes)
        if self.cfg.self_supervised.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self.loss = nn.CrossEntropyLoss()
        self.stage = "classification"
        task = "multiclass"
        average = "macro"
        metric_params = {"task":task, 
                         "num_classes": 10, 
                        "average": average, 
                        "top_k": 1}
        ### Metric objects for calculating and averaging accuracy across batches
        self.accuracy = Accuracy(**metric_params)
        self.f1 = F1Score(**metric_params)
        self.configure_optimizers()
    
    def training_step(self, batch, batch_idx):
        if self.stage == "self-supervised":
            (x1, x2), labels = batch
            z1 = self.model(x1)
            z2 = self.model(x2)
            loss = self.loss(z1, z2, self.current_epoch)
            self.log('contrastive_loss', 
                    loss, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
        else: 
            x, y = batch
            logits = self.forward(x)
            loss = self.loss(logits, y)
            self.log('classif_loss', 
                    loss, 
                    on_step=True,
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.stage == "self-supervised":
            (x1, x2), labels = batch
            z1 = self.model(x1)
            z2 = self.model(x2)
            loss = self.loss(z1, z2, self.current_epoch)
            self.log('contrastive_loss', 
                    loss, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
        else: 
            x, y = batch
            logits = self.forward(x)
            loss = self.loss(logits, y)
            self.log('val_accuracy',
                self.accuracy(logits, 
                              y),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True)
            self.log('val_f1',
                self.f1(logits, y),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True)
            self.log('classif_loss', 
                    loss, 
                    on_step=True,
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log('test_accuracy',
                self.accuracy(logits, 
                              y),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True)
        
        self.log('test_f1',
                self.f1(logits, y),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True)

    def configure_optimizers(self):
        if self.stage == "self-supervised":
            max_epochs = int(self.cfg.epochs)
            param_groups = define_param_groups(self.model, self.cfg.self_supervised.weight_decay, 'adam')
            lr = self.cfg.self_supervised.lr
            optimizer = Adam(param_groups, 
                            lr=lr, 
                            weight_decay=self.cfg.self_supervised.weight_decay)
            scheduler = CosineWarmupScheduler(optimizer, 
                            epoch_warmup=10, 
                            max_epoch=max_epochs,
                            min_lr=0.0)
            configuration = {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,}
        else:
            param_groups = define_param_groups(self.classifier, self.cfg.supervised.weight_decay, 'adam')
            lr = self.cfg.supervised.lr
            optimizer = Adam(param_groups, 
                            lr=lr, 
                            weight_decay=self.cfg.supervised.weight_decay)
            configuration = {
                "optimizer": optimizer,
                }
        return configuration


class Encoder(nn.Module):
   def __init__(self, config, model=None, mlp_dim=512):
       super().__init__()
       embedding_size = config.embedding_size
       self.backbone = default(model, models.resnet18(pretrained=False, num_classes=config.embedding_size))
       mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
       print('Dim MLP input:',mlp_dim)
       self.backbone.fc = nn.Identity()
       self.projection = nn.Sequential(
           nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
           nn.BatchNorm1d(mlp_dim),
           nn.ReLU(),
           nn.Linear(in_features=mlp_dim, out_features=embedding_size),
           nn.BatchNorm1d(embedding_size),)

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
        logit_scale = torch.ones([]) * self.cfg.self_supervised.temperature
        self.temperature = nn.Parameter(logit_scale)

    def forward(self, x_1, x_2, current_epoch=0):
        x_1 = torch.nn.functional.normalize(x_1, dim=-1, p=2)
        x_2 = torch.nn.functional.normalize(x_2, dim=-1, p=2)
        labels = torch.arange(x_1.shape[0], device=x_1.device) # entries on the diagonal
        similarity = einsum('i d, j d -> i j', x_1, x_2) / self.temperature
        loss_1 = torch.nn.functional.cross_entropy(similarity, labels) 
        loss_2 = torch.nn.functional.cross_entropy(similarity.T, labels) 
        return (loss_1 + loss_2) / 2.0
    

class vicREG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pass
    
    def forward(self, x, y):
        pass

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