import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

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
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.cov_coef = cfg['self_supervised']['cov_coef']
        self.batch_size = cfg['batch_size']
        self.num_features = cfg['embedding_size']

    def forward(self, x_1, x_2, *args, **kwargs):
        repr_loss = self.mse(x_1, x_2)

        x_1 = x_1 - x_1.mean(dim=0)
        x_2 = x_2 - x_2.mean(dim=0)

        std_x = torch.sqrt(x_1.var(dim=0) + 0.0001) # S(x_1, eps)
        std_y = torch.sqrt(x_2.var(dim=0) + 0.0001) # S(x_2, eps)
        # hinge loss 
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2 

        cov_x_1 = (x_1.T @ x_1) / (self.batch_size - 1) # att batch size is required
        cov_x_2 = (x_2.T @ x_2) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x_1).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_x_2).pow_(2).sum().div(self.num_features)

        loss = (repr_loss
                + std_loss
                + self.cov_coef * cov_loss)
        return loss

def off_diagonal(x):
    """Return off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()