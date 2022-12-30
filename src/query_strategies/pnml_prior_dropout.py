import logging

import torch
import wandb

from .pnml_dropout import DropoutPnml

logger = logging.getLogger(__name__)


class DropoutPnmlPrior(DropoutPnml):
    def __init__(
        self,
        n_drop: int = 10,
        query_batch_size=256,
        unlabeled_pool_size=256,
        test_set_size: int = 512,
    ):
        self.eps = 1e-6
        self.n_drop = n_drop
        self.batch_size = query_batch_size
        self.unlabeled_pool_size = unlabeled_pool_size
        self.test_set_size = test_set_size
        self.temperature = 20.0
        self.is_calibrate = False

    def regert_based_selection(
        self, mean_regrets, idx_candidates, dataset, n, candidate_probs
    ):

        regrets = candidate_probs * mean_regrets
        max_y_regret, max_y_idx = regrets.max(axis=-1)
        min_x_regret, min_x_idx = max_y_regret.sort(descending=False, axis=-1)
        choesn_idxs = idx_candidates[min_x_idx[:n]]

        # For debug:
        unlabled_labels = dataset.Y_train
        logger.info(f"Sorted true label {unlabled_labels[idx_candidates[min_x_idx]]}")
        logger.info(f"{min_x_regret=}")
        logger.info(f"{max_y_idx=}")
        logger.info(f"{unlabled_labels[choesn_idxs]=}")

        wandb.log(
            {
                "regret_of_chosen": min_x_regret[0].item(),
                "max_yn_value": max_y_idx[0].item(),
                "max_yn_mean_value": max_y_idx.float().mean().item(),
            }
        )

        return choesn_idxs

    def query(self, n, net, dataset):
        torch.set_grad_enabled(False)

        # Datasets
        candidate_loader, val_loader, test_loader = self.build_dataloaders(dataset)

        if self.is_calibrate:
            self.temperature = net.calibrate(val_loader)
        wandb.log({"temperature": self.temperature})
        logger.info(f"Temperature: {self.temperature}")

        candidate_probs = self.dataloader_inference(net, candidate_loader)

        # Inference
        regrets, candidate_idxs = self.iterate_candidate_loader(
            net, candidate_loader, test_loader
        )

        candidate_probs = self.dataloader_inference(net, candidate_loader)
        choesn_idxs = self.regert_based_selection(
            regrets, candidate_idxs, dataset, n, candidate_probs
        )

        candidate_max_probs = torch.round(candidate_probs.max(-1)[0], decimals=3)
        logger.info(f"candidate_max_probs={candidate_max_probs}")
        torch.set_grad_enabled(True)
        return choesn_idxs
