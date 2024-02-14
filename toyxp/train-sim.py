import torch
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
    name = f"{cfg.prefix}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    model = SimCLR_pl(cfg, 
                      model=resnet18(pretrained=False), 
                      feat_dim=512)
    transform = Augment(cfg.img_size)
    data_loader = get_stl_dataloader(root=cfg.paths.data, 
                                     batch_size=cfg.batch_size, 
                                     transform=transform)
    logger = (WandbLogger(name=name,
                        dir=cfg.paths.logs,
                        project=cfg.logger.project,
                        log_model=True) 
                        if cfg.log else None)

    ### Self-supervised 
    if cfg.self_supervised.pretrained:
        # accumulator = GradientAccumulationScheduler(scheduling={0: cfg.self_supervised.gradient_accumulation_steps})
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
                                     split="train")
    data_loader_test = get_stl_dataloader(root=cfg.paths.data, 
                                        batch_size=cfg.supervised.batch_size, 
                                        transform=transform.test_transform,
                                        split='test')
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