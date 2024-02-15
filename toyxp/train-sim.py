import torch
import os
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from torchvision.models import  resnet18
from omegaconf import DictConfig
import pyrootutils
import hydra
from pytorch_lightning.loggers import WandbLogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True)

from model import SimCLR_pl
from data_utils import Augment, get_stl_dataloader

@hydra.main(
    version_base="1.2",
    config_path=root / "config",
    config_name="cfg-sim.yaml",)
def train(cfg: DictConfig):
    if cfg.logger.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ['WANDB_DIR'] = cfg.paths.logs

    name = f"{cfg.prefix}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    model = SimCLR_pl(cfg, 
                      model=resnet18(pretrained=False), 
                      feat_dim=512)
    transform = Augment(cfg.img_size)
    data_loader = get_stl_dataloader(root=cfg.paths.data, 
                                     batch_size=cfg.batch_size, 
                                     transform=transform,
                                     num_workers=cfg.num_workers,)
    logger = (WandbLogger(name=name,
                        id=name,
                        project=cfg.logger.project,
                        log_model=True) 
                        if cfg.log else None)

    ### Self-supervised 
    if cfg.self_supervised.pretrained:
        trainer = Trainer(
                        logger=logger,
                        accelerator=cfg.trainer.accelerator,
                        overfit_batches=cfg.overfit_batches,
                        gpus=cfg.trainer.gpu,
                        max_epochs=cfg.self_supervised.epochs)
        trainer.fit(model, 
                    data_loader['train'], 
                    data_loader['val'])

    ### Linear evaluation 
    model.make_classifier()
    data_loader = get_stl_dataloader(root=cfg.paths.data, 
                                     batch_size=cfg.supervised.batch_size, 
                                     transform=transform.test_transform,
                                     split="train",
                                     num_workers=cfg.num_workers)
    data_loader_test = get_stl_dataloader(root=cfg.paths.data, 
                                        batch_size=cfg.supervised.batch_size, 
                                        transform=transform.test_transform,
                                        split='test',
                                        num_workers=cfg.num_workers)
    trainer_supervised = Trainer(callbacks=[],
                    logger=logger,
                    accelerator=cfg.trainer.accelerator,
                    overfit_batches=cfg.overfit_batches,
                    gpus=cfg.trainer.gpu,
                    max_epochs=cfg.supervised.epochs)
    trainer_supervised.fit(model, 
                data_loader['train'], 
                data_loader['val'])
    trainer_supervised.test(model, 
                 data_loader_test)
    
    if cfg.log:
        logger.finalize(status="success")


if __name__ == "__main__":
    train()