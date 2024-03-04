import torch
import os
from datetime import datetime
from pytorch_lightning import Trainer
from torchvision.models import  resnet18
from omegaconf import DictConfig
import pyrootutils
import hydra
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True)

from model import SimCLR_pl
from data_utils import Augment, get_stl_dataloader
from src.utils.helpers import SizeDatamodule

@hydra.main(
    version_base="1.2",
    config_path=root / "config",
    config_name="cfg-sim.yaml",)
def train(cfg: DictConfig):
    if cfg.logger.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ['WANDB_DIR'] = cfg.paths.logs
    callbacks = [SizeDatamodule()]
    name = f"{cfg.prefix}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    model = SimCLR_pl(cfg, 
                      model=resnet18(pretrained=False), 
                      feat_dim=512)
    transform = Augment(cfg.img_size)
    data_loader = get_stl_dataloader(cfg=cfg,
                                     batch_size=cfg.batch_size, 
                                     transform=transform)
    logger = (WandbLogger(name=name,
                        id=name,
                        project=cfg.logger.project,
                        log_model=True) 
                        if cfg.log else None)
    if cfg.log: 
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    progress_bar = False if cfg.trainer.___config_name___ == "jz" else True 
    ### Self-supervised 
    if cfg.self_supervised.pretrained:
        trainer = Trainer(logger=logger,
                        callbacks=callbacks,
                        enable_progress_bar=progress_bar,
                        overfit_batches=cfg.overfit_batches,
                        accelerator=cfg.trainer.accelerator,
                        gpus=cfg.trainer.devices,
                        max_epochs=cfg.self_supervised.epochs)
        trainer.fit(model, 
                    data_loader['train'], 
                    data_loader['val'])

    ### Linear evaluation 
    if cfg.classification:
        model.stage = "classification"
        callbacks = [SizeDatamodule()]
        if cfg.log:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        # if cfg.ckpt:
        #    callbacks.append(ModelCheckpoint(save_last=False, 
        #                                     dirpath=f"checkpoints", 
        #                                     filename=name))
        data_loader = get_stl_dataloader(cfg=cfg,
                                        batch_size=cfg.supervised.batch_size, 
                                        transform=transform.test_transform,
                                        split="train")
        
        data_loader_test = get_stl_dataloader(cfg=cfg,
                                            batch_size=cfg.supervised.batch_size, 
                                            transform=transform.test_transform,
                                            split='test')
        
        trainer_supervised = Trainer(callbacks=callbacks,
                                    enable_progress_bar=progress_bar,
                                    logger=logger,
                                    overfit_batches=cfg.overfit_batches,
                                    accelerator=cfg.trainer.accelerator,
                                    gpus=cfg.trainer.devices,
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