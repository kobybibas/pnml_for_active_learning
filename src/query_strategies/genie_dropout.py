import logging

import torch
import wandb

from .pnml_dropout import DropoutPnml

logger = logging.getLogger(__name__)


class DropoutGenie(DropoutPnml):
    """
    Genie that knows the true labels of the unlalbled pool
    """

    def regert_based_selection(self, mean_regrets, idx_candidates, dataset, n):

        assert dataset.labeled_idxs[idx_candidates].sum() == 0
        true_labels = dataset.Y_train[idx_candidates].to(mean_regrets.device)
        regret_based_on_true_y = torch.hstack(
            [
                mean_regrets[i, true_label.item()]
                for i, true_label in enumerate(true_labels)
            ]
        )
        if False:

            min_x_regret, min_x_idx = regret_based_on_true_y.sort(
                descending=False, axis=-1
            )
            wandb.log({"regret_of_chosen": min_x_regret[0].item()})
        else:
            # Take from the most scarce label
            most_scarce_label = (
                dataset.Y_train[dataset.labeled_idxs].bincount().argmin()
            )
            mask = true_labels == most_scarce_label.to(mean_regrets.device)
            regret_based_on_true_y = regret_based_on_true_y[mask]
            idx_candidates = idx_candidates[mask]
        min_x_regret, min_x_idx = regret_based_on_true_y.sort(descending=False, axis=-1)
        wandb.log({"regret_of_chosen": min_x_regret[0].item()})

        return idx_candidates[min_x_idx[:n]]
