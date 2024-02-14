import torch
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
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
from data_utils import Augment, get_stl_dataloader

@hydra.main(
    version_base="1.2",
    config_path=root / "toyxp",
    config_name="cfg-sim.yaml",)
def train(cfg: DictConfig):
    name = "simclr" + cfg.prefix
    model = SimCLR_pl(cfg, 
                      model=resnet18(pretrained=False), 
                      feat_dim=512)
    transform = Augment(cfg.img_size)
    data_loader = get_stl_dataloader(root=cfg.stl10, 
                                     batch_size=cfg.batch_size, 
                                     transform=transform)
    ### Self-supervised steps
    accumulator = GradientAccumulationScheduler(scheduling={0: cfg.training.gradient_accumulation_steps})
    trainer = Trainer(callbacks=[accumulator],
                      overfit_batches=cfg.overfit_batches,
                     gpus=cfg.gpu,
                     max_epochs=cfg.epochs)
    trainer.fit(model, data_loader['train'], data_loader['val'])

    ### Linear evaluation 
    model.make_classifier()
    data_loader = get_stl_dataloader(root=cfg.stl10, 
                                     batch_size=cfg.batch_size, 
                                     transform=transform.test_transform,
                                     split="train")
    data_loader_test = get_stl_dataloader(root=cfg.stl10, 
                                        batch_size=cfg.batch_size, 
                                        transform=transform.test_transform,
                                        split='test')
    trainer.fit(model, data_loader['train'], data_loader['val'])
    trainer.test(model, data_loader_test)


if __name__ == "__main__":
    train()