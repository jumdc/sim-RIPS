import torch
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import  resnet18


def train():
    model = SimCLR_pl(train_config, model=resnet18(pretrained=False), feat_dim=512)

    transform = Augment(train_config.img_size)