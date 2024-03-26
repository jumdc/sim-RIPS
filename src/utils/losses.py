import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance
import logging

class symInfoNCE(nn.Module):
    """symmetric infoNCE loss, aka same scheme than in ImageBind"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        logit_scale = torch.ones([]) * self.cfg.self_supervised.temperature
        self.temperature = nn.Parameter(logit_scale)

    def forward(self, x_1, x_2, *args, **kwargs):
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
        self.std_coef = cfg['self_supervised']['std_coef']
        self.batch_size = cfg['batch_size']
        self.num_features = cfg['embedding_size']
        logging.info(f"vicREG loss with cov_coef: {self.cov_coef} and std_coef: {self.std_coef}")
        if cfg['self_supervised']['dtm_reg']:
            logging.info("vicREG loss with DTM regularization")
            self.dtm = DTMLoss(cfg)
            

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
                + self.std_coef * std_loss
                + self.cov_coef * cov_loss)
        if hasattr(self, 'dtm'):
            loss += self.dtm(x_1, x_2)
        return loss

def off_diagonal(x):
    """Return off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class DTMLoss(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """DTM-based loss, as in Deep Topological Metric Learning"""
        super().__init__(*args, **kwargs)
        self.k = cfg['dtm']['k']
        self.MSE = nn.MSELoss()

    def forward(self, x_1, x_2, *args, **kwargs):
        """forward"""
        dist_x1 = torch.cdist(x_1, x_1, p=2)
        dist_x2 = torch.cdist(x_2, x_2, p=2)

        topk_x1 = torch.sum(torch.topk(dist_x1, k=self.k+1, largest=False)[0], 
                            dim=1)
        topk_x2 = torch.sum(torch.topk(dist_x2, k=self.k+1, largest=False)[0], 
                            dim=1)
        loss = self.MSE(topk_x1, topk_x2)
        return loss
    


class TopologicalLoss(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()
        self.mse = nn.MSELoss()
        print(cfg["topological"])
        self.rips = VietorisRipsComplex(dim=cfg['topological']['max_dim'])
        self.batch_size = cfg['batch_size'] # normalize as in Topological AutoEncoders
        self.distance = WassersteinDistance(p=2) # euclidean ?
        self.w_l2 = cfg['topological']['w_l2']
        self.w_topo = cfg['topological']['w_topo']
        self.logger = logger

    def forward(self, x_1, x_2, *args, **kwargs):
        representation_loss = self.mse(x_1, x_2)
        pi_1 = self.rips(x_1)
        pi_2 = self.rips(x_2)
        if (len(args) > 0 
            and args[0]
            and isinstance(self.logger, WandbLogger)):
            fig = plot_diagram(pi_1, pi_2)
            self.logger.experiment.log({f"persistent_diagram": wandb.Image(fig)})
        
        topological_loss = (self.distance(pi_1, pi_2) + self.distance(pi_2, pi_1)) / 2 * self.batch_size

        return self.w_l2 * representation_loss + self.w_topo * topological_loss
    

def plot_diagram(pi, pi_2):
    fig, axs = plt.subplots(constrained_layout=True, 
                            ncols=2,
                            figsize=(15, 10))
    maxi_x, maxi_y = 0, 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(pi)))
    for dim in range(len(pi)):
        # diag 1
        diag = pi[dim].diagram.detach().cpu().numpy()
        diag_2 = pi_2[dim].diagram.detach().cpu().numpy()
        if diag.shape[0] > 0:
            inf_idx = np.where(np.isinf(diag[:,1]))
            birth_inf = diag[inf_idx,0]
            diag = np.delete(diag, 
                             inf_idx, 
                             axis=0)
            if diag.shape[0] > 0: # check if  there are other than inf vlaues
                maxi_x_1 = np.max(diag[:,0])
                maxi_y_1 = np.max(diag[:,1])
            else : 
                maxi_x_1 = 0
                maxi_y_1 = 0
        else: 
            diag = np.array([])
            maxi_x_1 = 0
            maxi_y_1 = 0
        # diag 2
        if diag_2.shape[0] > 0:
            inf_idx_2 = np.where(np.isinf(diag_2[:,1]))
            birth_inf_2 = diag_2[inf_idx,0]
            diag_2 = np.delete(diag_2, inf_idx_2, axis=0)
            if diag_2.shape[0] > 0: # check if there are other than inf values 
                maxi_x_2 = np.max(diag_2[:,0])
                maxi_y_2 = np.max(diag_2[:,1])
            else: 
                maxi_x_2 = 0
                maxi_y_2 = 0
        else: 
            diag_2 = np.array([])
            maxi_x_2 = 0
            maxi_y_2 = 0
        # max
        maxi_x = max(maxi_x, maxi_x_1, maxi_x_2)
        maxi_y = max(maxi_y, maxi_y_1, maxi_y_2)
        if diag.shape[0] > 0:
            axs[0].scatter(diag[:, 0], 
                        diag[:, 1], 
                            c=colors[dim],
                            marker="x",
                        label=f"$H_{dim}$ - number of points: {diag.shape[0]}")
            axs[0].scatter(birth_inf, 
                    np.repeat(maxi_y, birth_inf.shape[1]),
                    marker="o",
                    s=30,
                    c=colors[dim],
                    label=f"$H_{dim}$ - inf")
        if diag_2.shape[0] > 0:
            axs[1].scatter(diag_2[:, 0], 
                        diag_2[:, 1], 
                            marker="x",
                            c=colors[dim],
                        label=f"$H_{dim}$ - number of points: {diag_2.shape[0]}")
            axs[1].scatter(birth_inf_2, 
                    np.repeat(maxi_y, birth_inf_2.shape[1]),
                    marker="o",
                    s=30,
                    c=colors[dim],
                    label=f"$H_{dim}$ - inf")

    axs[0].set_title("Persistent Diagram - view 1")
    axs[0].set_xlim(0, max(maxi_x, maxi_y))
    axs[0].set_ylim(0, max(maxi_x, maxi_y))
    axs[0].plot([0,  max(maxi_x, maxi_y)],[0,  max(maxi_x, maxi_y)], c="lightgrey") # diagonal
    
    axs[0].set_xlabel("Birth")
    axs[0].set_ylabel("Death")
    axs[0].legend()

    axs[1].set_title("Persistent Diagram - view 2")
    axs[1].set_xlim(0, max(maxi_x, maxi_y))
    axs[1].set_ylim(0, max(maxi_x, maxi_y))
    axs[1].plot([0, max(maxi_x, maxi_y)],[0, max(maxi_x, maxi_y)], c="lightgrey") # diagonal

    axs[1].set_xlabel("Birth")
    axs[1].set_ylabel("Death")
    axs[1].legend()

    fig.canvas.draw_idle()
    pd = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    pd = pd.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")
    pd = np.expand_dims(pd, axis=0) 
    return pd
    