import logging
import os
import os.path as osp
import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from data import get_dataloaders
from lit_utils import LitClassifier
from utils import get_dataset, get_net, get_strategy

logger = logging.getLogger(__name__)


def initalize_trainer(
    cfg: DictConfig,
    wandb_logger,
    enable_progress_bar: bool = False,
    is_lr_monitor: bool = False,
    is_swa: bool = False,
    out_dir: str = None,
):
    callbacks = []
    if is_swa is False:
        callbacks.append(
            EarlyStopping(
                monitor="acc/val",
                mode="max",
                patience=cfg.early_stopping_patience,
            )
        )
    if is_swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=0.0, annealing_epochs=5, swa_lrs=cfg.lr
            )
        )
    if is_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs if is_swa is False else 10,
        min_epochs=cfg.min_epochs if is_swa is False else 10,
        default_root_dir=out_dir,
        enable_checkpointing=False,
        gpus=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=1 if enable_progress_bar else 100,
        callbacks=callbacks,
    )
    return trainer


@hydra.main(version_base="1.2", config_path="../configs/", config_name="main")
def execute_active_learning(cfg: DictConfig):
    t0 = time.time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(cfg.seed)
    wandb_group_id = os.getenv("WANDB_GROUP_ID")

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type=f"{cfg.dataset_name}_{cfg.strategy_name}_{wandb_group_id}",
        name=name,
    )
    wandb_logger = WandbLogger(experimnet=wandb.run)
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Load dataset
    dataset = get_dataset(cfg.dataset_name, cfg.data_dir, cfg.validation_set_size)
    net = get_net(cfg.dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sampling strategy
    strategy = get_strategy(
        cfg.strategy_name,
        n_drop=cfg.n_drop,
        unlabeled_batch_size=cfg.unlabeled_batch_size,
        unlabeled_pool_size=cfg.unlabeled_pool_size,
        test_batch_size=cfg.test_batch_size,
        test_set_size=cfg.test_set_size,
    )

    # Active learning
    dataset.initialize_labels(cfg.n_init_labeled)
    for rd in range(cfg.n_round):
        t1 = time.time()

        lit_h = LitClassifier(net, cfg, device)
        trainer = initalize_trainer(
            cfg,
            wandb_logger,
            enable_progress_bar=rd == 0,
            is_lr_monitor=rd == 0,
            is_swa=False,
            out_dir=out_dir,
        )

        # Execute training
        train_loader, val_loader, _ = get_dataloaders(dataset, cfg.batch_size)
        trainer.fit(lit_h, train_loader, val_loader)

        if cfg.strategy_name == "SwaPnml":
            logger.info("SwaPnml training")
            # Extra training epochs for SWA
            trainer = initalize_trainer(
                cfg,
                wandb_logger,
                enable_progress_bar=rd == 0,
                is_lr_monitor=rd == 0,
                is_swa=True,
                out_dir=out_dir,
            )
            trainer.fit(lit_h, train_loader, val_loader)

        lit_h = lit_h.to(device).float()
        lit_h = lit_h.eval()

        # Calculate performance
        training_labels = dataset.get_labeled_labels().cpu().numpy()
        preds, probs = lit_h.predict(dataset.get_test_data())
        test_acc = dataset.cal_test_acc(preds)
        test_loss = dataset.cal_test_loss(probs)

        # Query and update labels
        query_idxs = strategy.query(cfg.n_query, lit_h, dataset)
        strategy.update(dataset, query_idxs)

        round_time = time.time() - t1
        logger.info(
            f"[{rd}/{cfg.n_round-1}] Testing acc={test_acc:.3f}. Train size={len(training_labels)} in {round_time:.2f} sec"
        )

        wandb.log(
            {
                "active_learning_round": rd,
                "training_set_size": len(training_labels),
                "test_acc": test_acc,
                "test_loss": test_loss,
                "round_time_sec": round_time,
            }
            | {
                f"label_{label}_ratio": (training_labels == label).sum()
                / len(training_labels)
                for label in np.arange(0, 10, 1)
            }
        )

    logger.info(f"Finish in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    execute_active_learning()
