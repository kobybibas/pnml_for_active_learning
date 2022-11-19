import logging
import time
from time import time

import torch
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np


from .strategy import Strategy

logger = logging.getLogger(__name__)


class DropoutPnml(Strategy):
    def __init__(
        self,
        n_drop: int = 10,
        unlabeled_batch_size=256,
        unlabeled_pool_size=256,
        test_batch_size=256,
        test_set_size: int = 512,
    ):
        self.eps = 1e-6
        self.n_drop = n_drop
        self.unlabeled_batch_size = unlabeled_batch_size
        self.unlabeled_pool_size = unlabeled_pool_size
        self.test_batch_size = test_batch_size
        self.test_set_size = test_set_size

    def model_inference(self, net, x_candidates, x_test):
        num_candidates, num_test = len(x_candidates), len(x_test)

        # Inference
        preds, _ = net(torch.vstack((x_candidates, x_test)))
        num_labels = preds.size(1)
        probs = torch.softmax(preds, -1)
        log_probs = torch.log_softmax(preds, -1)
        log_probs_candidates, log_probs_test = (
            log_probs[:num_candidates],
            log_probs[num_candidates:],
        )
        probs_test = probs[num_candidates:]

        # Calculate log_probs_candidates + log_probs_test
        a, b = torch.meshgrid(log_probs_candidates.flatten(), log_probs_test.flatten())
        scores = (a + b).reshape(num_candidates, num_labels, num_test, num_labels)

        return probs_test, scores

    def query(self, n, net, dataset):
        torch.set_grad_enabled(False)

        # Datasets
        candidate_idx_original, candidate_set = dataset.get_unlabeled_data()
        idx_subset = torch.randperm(len(candidate_set))[: self.unlabeled_pool_size]
        candidate_subset = Subset(candidate_set, idx_subset)

        test_set = dataset.get_test_data()
        idx_subset = torch.randperm(len(test_set))[: self.test_set_size]
        test_subset = Subset(test_set, idx_subset)

        # Dataloaders
        candidate_loader = DataLoader(
            candidate_subset,
            batch_size=self.unlabeled_batch_size,
            num_workers=0,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_subset, batch_size=self.test_batch_size, num_workers=0, shuffle=False
        )

        # Inference
        net.train()
        mean_regrets, idx_candidates = [], []
        for x_candidates, _, idx_candidates_batch in candidate_loader:
            regrets = []
            for x_test, _, _ in test_loader:
                model_scores, model_probs_test = [], []
                for _ in range(self.n_drop):
                    num_candidates, num_test = len(x_candidates), len(x_test)
                    probs_test, scores = self.model_inference(net, x_candidates, x_test)
                    num_labels = probs_test.size(1)
                    model_scores.append(scores.cpu())
                    model_probs_test.append(probs_test.cpu())

                model_scores = torch.stack(model_scores, -1)
                model_probs_test = torch.stack(model_probs_test, -1)

                # Duplicate to fit model_scores
                model_probs_test = (
                    model_probs_test.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_candidates, num_labels, 1, 1, 1)
                )

                # Calculate the model that maxzimies log_probs_candidates + log_probs_test
                _, model_chosen = model_scores.max(axis=-1)

                # Best model prediction probability
                max_probs = torch.gather(
                    model_probs_test, -1, model_chosen.unsqueeze(-1)
                ).squeeze()

                # Regret for each test sample
                regrets_i = max_probs.sum(axis=-1).log()
                regrets.append(regrets_i)

            # Average regret over the test set
            regrets = torch.cat(regrets, -1).mean(axis=-1)
            mean_regrets.append(regrets)
            idx_candidates.append(idx_candidates_batch)

        # Verify all candidates indecies are indeed unlabeled.
        idx_candidates = torch.hstack(idx_candidates)
        assert (
            torch.isin(idx_candidates, torch.from_numpy(candidate_idx_original))
            .min()
            .item()
        )

        mean_regrets = torch.vstack(mean_regrets)
        max_y_regret, max_y_idx = mean_regrets.max(axis=-1)
        min_x_regret, min_x_idx = max_y_regret.sort(descending=False, axis=-1)

        wandb.log(
            {"regret_of_chosen": min_x_regret, "max_yn_value": max_y_idx,}
        )

        torch.set_grad_enabled(True)

        return idx_candidates[min_x_idx[:n]]

