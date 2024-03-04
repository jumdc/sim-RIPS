import torch
import numpy as np
from scipy.linalg import svd
from matplotlib import pyplot as plt
import wandb

def compute_metrics_contrastive(logger,
                                outputs,
                                prefix="",
                                plot=False,
                                key_1: str = "view_1"):
    if plot: 
        x_representation = torch.cat([x[key_1] for x in outputs]).cpu().detach()
        cov_X = np.cov(x_representation.T)
        singular_values = svd(cov_X, compute_uv=False)
        cum_singular_values = np.cumsum(singular_values)
        total = sum(singular_values)
        cum_singular_values = cum_singular_values / total
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(cum_singular_values, c="#f94144")
        ax.set_ylabel("Cumulative singular values")
        ax.set_xlabel("Singular Value Rank Index")
        ax.set_title("Cumulative explained variance.")

        fig.canvas.draw_idle()
        pd = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        pd = pd.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        pd = np.expand_dims(pd, axis=0) 
        if logger is not None:
            logger.experiment.log({f"{prefix}-variance": wandb.Image(fig)})