import logging
import os
import os.path as osp
import time
from glob import glob

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid

from data import get_dataloaders
from lit_utils import LitClassifier, initalize_trainer
from utils import get_dataset, get_net, get_strategy

logger = logging.getLogger(__name__)


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
        query_batch_size=cfg.query_batch_size,
        unlabeled_pool_size=cfg.unlabeled_pool_size,
        test_set_size=cfg.test_set_size,
    )

    # Active learning
    dataset.initialize_labels(cfg.n_init_labeled)
    for rd in range(cfg.n_round):
        t1 = time.time()
        is_debug = rd == 0

        class_weight = dataset.get_labeled_labels().cpu()
        class_weight = class_weight[class_weight != -1]
        class_weight = 1 / torch.bincount(class_weight)
        lit_h = LitClassifier(net, cfg, device, class_weight)
        trainer, checkpoint_callback = initalize_trainer(
            cfg,
            wandb_logger,
            enable_progress_bar=is_debug,
            is_lr_monitor=is_debug,
            out_dir=out_dir,
        )

        # Execute training
        train_loader, val_loader, _ = get_dataloaders(dataset, cfg.batch_size)
        trainer.fit(lit_h, train_loader, val_loader)
        lit_h = LitClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
        [os.remove(file) for file in glob(f"{out_dir}/*.ckpt")]
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
                for label in np.arange(training_labels.min(), training_labels.max(), 1)
            }
        )
        if True:
            imgs = dataset.X_train_org[query_idxs]
            image_array = make_grid(
                imgs,
                nrow=1,
            )
            images = wandb.Image(image_array)
            wandb.log({"Queried images": images})

    logger.info(f"Finish in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    execute_active_learning()
