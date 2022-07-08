import logging
import os
import os.path as osp
import time

import hydra
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from lit_utils import LitClassifier
from utils import get_dataset, get_net, get_strategy
from data import get_dataloaders

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="main")
def execute_active_learning(cfg: DictConfig):
    t0 = time.time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type=f"{cfg.dataset_name}_{cfg.strategy_name}",
        name=name,
    )
    wandb_logger = WandbLogger(experimnet=wandb.run)
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Load dataset
    dataset = get_dataset(
        cfg.dataset_name, cfg.training_set_size, cfg.validation_set_size, cfg.data_dir
    )

    # Architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = get_net(cfg.dataset_name)

    # Sampling strategy
    strategy = get_strategy(cfg.strategy_name)
    if cfg.strategy_name == "SingleLayerPnml":
        strategy.unlabeled_batch_size = cfg.SingleLayerPnml.unlabeled_batch_size
        strategy.unlabeled_pool_size = cfg.SingleLayerPnml.unlabeled_pool_size

    # Active learning
    dataset.initialize_labels(cfg.n_init_labeled)
    for rd in range(cfg.n_round):
        t1 = time.time()

        # Train
        lit_h = LitClassifier(net, cfg, device)
        trainer = pl.Trainer(
            max_epochs=cfg.epochs_max,
            min_epochs=cfg.epochs_min,
            default_root_dir=out_dir,
            enable_checkpointing=False,
            gpus=1 if torch.cuda.is_available() else None,
            logger=wandb_logger,
            precision=16,
            num_sanity_val_steps=0,
            enable_progress_bar=rd == 0,
            enable_model_summary=rd == 0,
            callbacks=[pl.callbacks.LearningRateMonitor(),],
        )

        # Execute training
        train_loader, val_loader = get_dataloaders(
            dataset, cfg.batch_size, cfg.batch_size_test
        )
        trainer.fit(lit_h, train_loader, val_loader)
        lit_h.clf = lit_h.clf.half()
        lit_h = lit_h.to(device)

        # Calculate performance
        training_labels = dataset.get_labeled_labels().cpu().numpy()
        preds, probs = lit_h.predict(dataset.get_test_data())
        test_acc = dataset.cal_test_acc(preds)
        test_loss = dataset.cal_test_loss(probs)

        round_time = time.time() - t1
        logger.info(
            f"[{rd}/{cfg.n_round-1}] Testing [acc loss]=[{test_acc:.3f} {test_loss:.2f}]. Train size={len(training_labels)}"
        )

        wandb.log(
            {
                "active_learning_round": rd,
                "training_set_size": len(training_labels),
                "test_acc": test_acc,
                "test_loss": test_loss,
                "round_time_sec": round_time,
            }
        )
        wandb.log(
            {
                f"label_{label}_ratio": (training_labels == label).sum()
                / len(training_labels)
                for label in np.arange(0, 10, 1)
            }
        )

        # Query and update labels
        t1 = time.time()
        query_idxs = strategy.query(cfg.n_query, lit_h, dataset)
        strategy.update(dataset, query_idxs)
        logger.info(f"\t{cfg.strategy_name} query in {time.time()-t1:.2f} sec")

        if False:
            img = dataset.X_train[query_idxs]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().numpy()
            img = PIL.Image.fromarray(img)
            wandb.log({"Image to label": wandb.Image(img)})

    logger.info(f"Finish in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    execute_active_learning()
