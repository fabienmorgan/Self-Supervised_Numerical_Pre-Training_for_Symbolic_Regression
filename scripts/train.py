import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from dataclasses import dataclass
from typing import Tuple
from ControllableNesymres.architectures.model import Model
from ControllableNesymres.architectures.data import DataModule
import hydra
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf



lr_monitor = LearningRateMonitor(logging_interval='step')

@hydra.main(config_name="config", version_base="1.2", config_path="")
def main(cfg):
    #seed_everything(9)
    train_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.train_path))
    benchmark_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.benchmark_path))
    data = DataModule(
        train_path,
        benchmark_path,
        cfg
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb_logger = WandbLogger(project="MMSR", entity="fabien-morgan", config=cfg_dict, log_model=True)

    cfg.inference.word2id = data.training_dataset.word2id
    cfg.inference.id2word = data.training_dataset.id2word
    cfg.inference.total_variables = data.training_dataset.total_variables
    
    model = Model(cfg=cfg)

    data.setup() # Ugly hack in order to create the mapper
    data.val_dataloader()
    model.mapper = data.mapper 
    model.metadata = data.training_dataset.metadata
    model.cfg.inference.id2word = data.training_dataset.id2word
    

    if cfg.host_system_config.resume_from_checkpoint:
        candidate_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.resume_from_checkpoint))
        # Check if the path is a file or a directory
        if candidate_path.is_file():
            is_folder = False
            checkpoint_absolut_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.resume_from_checkpoint)).parent
            logs_save_dir_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.resume_from_checkpoint)).parent.parent / "logs_dir"
        elif candidate_path.is_dir():
            is_folder = True
            checkpoint_absolut_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.resume_from_checkpoint))
            logs_save_dir_path = Path(hydra.utils.to_absolute_path(cfg.host_system_config.resume_from_checkpoint)).parent / "logs_dir"

        logger = pl_loggers.TensorBoardLogger(save_dir=logs_save_dir_path, sub_dir="logs/", name="", version="")
    else:
        logger = pl_loggers.TensorBoardLogger(save_dir="logs_dir/", sub_dir="logs/", name="", version="")
        
        
    checkpoint_dir_path = "exp_weights/"
    
    checkpoint_callback = ModelCheckpoint(
        #monitor="train_loss", #/dataloader_idx_0",
        dirpath=checkpoint_dir_path,                 
        filename=train_path.stem+"_log_"+"-{epoch:02d}-{val_loss:.2f}",
        mode="min",
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )

    if cfg.host_system_config.resume_from_checkpoint:
        print("Resuming from checkpoint")
        if is_folder:
            # Find the latest checkpoint
            checkpoints = list(checkpoint_absolut_path.glob("**/*.ckpt"))
            checkpoints.sort(key=os.path.getmtime)
            path_to_restart = checkpoints[-1]
        else:
            path_to_restart = Path(hydra.utils.to_absolute_path(cfg.host_system_config.resume_from_checkpoint))
    else:
        path_to_restart = None

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator=cfg.host_system_config.accelerator,
        devices=cfg.host_system_config.accelerator_devices,
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        precision=cfg.host_system_config.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        resume_from_checkpoint=path_to_restart,
        logger=wandb_logger,
    )

    wandb_logger.watch(model, log_freq=10)
    trainer.fit(model, data)

    wandb.finish()


if __name__ == "__main__":
    main()