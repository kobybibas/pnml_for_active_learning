import logging
import time
from time import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .pnml_dropout import DropoutPnml

logger = logging.getLogger(__name__)


class DropoutGenie(DropoutPnml):
    """
    Genie that knows the true labels of the unlalbled pool
    """

    def regert_best_selection(self, mean_regrets, idx_candidates, dataset, n):

        true_labels = dataset.Y_train[idx_candidates].to(mean_regrets.device)
        regret_based_on_true_y = torch.hstack(
            [
                mean_regrets[i, true_label.item()]
                for i, true_label in enumerate(true_labels)
            ]
        )
        min_x_regret, min_x_idx = regret_based_on_true_y.sort(descending=False, axis=-1)
        wandb.log("regret_of_chosen", min_x_regret[0].item())
        return idx_candidates[min_x_idx[:n]]
