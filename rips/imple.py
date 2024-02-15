import gudhi
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
# import tensorflow as tf

from gudhi.wasserstein import wasserstein_distance
from torch.optim.lr_scheduler import LambdaLR

def myloss(pts_1, pts):
    rips = gudhi.RipsComplex(points=pts, max_edge_length=0.5)
    rips_1 = gudhi.RipsComplex(points=pts_1, max_edge_length=0.5)
    # .5 because it is faster and, experimentally, the cycles remain smaller
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    i = st.flag_persistence_generators()
    if len(i[1]) > 0:
        i1 = torch.tensor(i[1][0])  # pytorch sometimes interprets it as a tuple otherwise
    else:
        i1 = torch.empty((0, 4), dtype=int)

    # for the seconds Point cloud
    st_1 = rips_1.create_simplex_tree(max_dimension=2)
    st_1.compute_persistence()
    i_1 = st_1.flag_persistence_generators()
    if len(i_1[1]) > 0:
        i1_1 = torch.tensor(i_1[1][0])
    else:
        i1_1 = torch.empty((0, 4), dtype=int)
    # Same as the finite part of st.persistence_intervals_in_dimension(1), but differentiable
    diag = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)
    diag1 = torch.norm(pts_1[i1_1[:, (0, 2)]] - pts_1[i1_1[:, (1, 3)]], dim=-1)

    wasser = wasserstein_distance(diag, diag1, order=1, enable_autodiff=True)
    return wasser

pts_1 = (torch.rand((200, 2)) * 2 - 1).requires_grad_()
for param in pts_1:
    param.requires_grad = False
pts = (torch.rand((200, 2)) * 2 - 1).requires_grad_()

opt = torch.optim.SGD([pts], lr=1)
scheduler = LambdaLR(opt,[lambda epoch: 10./(10+epoch)])
for idx in range(600):
    opt.zero_grad()
    myloss(pts_1, pts).backward()
    opt.step()
    scheduler.step()
    # Draw every 100 epochs
    if idx % 100 == 99:
        P = pts.detach().numpy()
        plt.scatter(P[:, 0], P[:, 1])
        plt.savefig("rips/figs/rips" + str(idx) + ".png")
        plt.close()


        