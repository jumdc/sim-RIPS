import torch
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import  resnet18
from omegaconf import DictConfig
import pyrootutils
import hydra

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True)

from model import SimCLR_pl
from data_utils import Augment

@hydra.main(
    version_base="1.2",
    config_path=root / "toy-xp",
    config_name="cfg-sim.yaml",)
def train(cfg: DictConfig):
    model = SimCLR_pl(cfg, 
                      model=resnet18(pretrained=False), 
                      feat_dim=512)
    transform = Augment(cfg.img_size)


if __name__ == "__main__":
    train()