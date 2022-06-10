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

from utils import get_dataset, get_net, get_strategy

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="main")
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
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    dataset = get_dataset(cfg.dataset_name, cfg.data_dir)  # Load dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = get_net(cfg.dataset_name, cfg[cfg.dataset_name], device)  # Load network
    strategy = get_strategy(cfg.strategy_name)(dataset, net)  # Load strategy

    # Start experiment
    dataset.initialize_labels(cfg.n_init_labeled)
    logger.info(f"Number of labeled pool: {cfg.n_init_labeled}")
    logger.info(f"Number of unlabeled pool: {dataset.n_pool-cfg.n_init_labeled}")
    logger.info(f"Number of testing pool: {dataset.n_test}")

    # Active learning
    for rd in range(cfg.n_round):
        logger.info(f"Round {rd}")
        t1 = time.time()

        if rd > 0:
            # Query and update labels
            t1 = time.time()
            query_idxs = strategy.query(cfg.n_query)
            strategy.update(query_idxs)
            logger.info(f"\t{cfg.strategy_name} query in {time.time()-t1:.2f} sec")

            img = strategy.dataset.X_train[query_idxs]
            wandb.log({"Image to label": wandb.Image(PIL.Image.fromarray(img.numpy()))})

        # Train
        strategy.train()

        # Calculate performance
        training_labels = strategy.dataset.get_labeled_labels()
        preds = strategy.predict(dataset.get_test_data())
        probs = strategy.predict_prob(dataset.get_test_data())
        test_acc = dataset.cal_test_acc(preds)
        test_loss = dataset.cal_test_loss(probs)
        training_labels_hist = np.histogram(
            training_labels.cpu().numpy(), bins=torch.arange(0, 10, 1), density=True,
        )
        round_time = t1 = time.time()
        logger.info(f"\ttraining_labels_hist: {training_labels_hist[0]}")
        logger.info(
            f"[{rd}/{cfg.n_round-1}] Testing [acc loss]=[{test_acc:.3f} {test_loss:.2f}]. training_set_size={len(training_labels)}"
        )

        wandb.log(
            {
                "active_learning_round": rd,
                "training_set_size": len(training_labels),
                "test_acc": test_acc,
                "test_loss": test_loss,
                "training_set_hist": wandb.Histogram(np_histogram=training_labels_hist),
                "round_time": round_time,
            }
        )

    logger.info(f"Finish in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    execute_active_learning()
