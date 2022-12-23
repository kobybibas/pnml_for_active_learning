import logging

import torch
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .strategy import Strategy

logger = logging.getLogger(__name__)


class DropoutPnml(Strategy):
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
        self.temperature = 1.0

    def build_dataloaders(self, dataset):
        # Candidate set
        _, candidate_set = dataset.get_unlabeled_data()
        candidate_idx_subset = torch.randperm(len(candidate_set))[
            : self.unlabeled_pool_size
        ]
        candidate_subset = Subset(candidate_set, candidate_idx_subset)

        # Test set
        test_set = dataset.get_test_data()
        test_idx_subset = torch.randperm(len(test_set))[: self.test_set_size]
        test_subset = Subset(test_set, test_idx_subset)

        # Dataloaders
        candidate_loader = DataLoader(
            candidate_subset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        val_loader = DataLoader(
            dataset.get_val_data(),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_subset, batch_size=self.batch_size, num_workers=0, shuffle=False
        )
        return candidate_loader, val_loader, test_loader

    def model_inference(self, net, x_candidates, x_test):
        num_candidates, num_test = len(x_candidates), len(x_test)

        # Inference
        device = net.device
        x_candidates, x_test = x_candidates.to(device), x_test.to(device)
        preds, _ = net(
            torch.vstack((x_candidates, x_test)), temperature=self.temperature
        )
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

    def calc_test_batch_regret(self, net, x_candidates, x_test):
        model_scores, model_test_probs = [], []
        for _ in range(self.n_drop):
            test_probs, scores = self.model_inference(net, x_candidates, x_test)
            model_scores.append(scores.cpu())
            model_test_probs.append(test_probs.cpu())

        model_scores = torch.stack(model_scores, -1)
        model_test_probs = torch.stack(model_test_probs, -1)
        return model_scores, model_test_probs

    def iterate_test_loader(self, net, test_loader, x_candidates):
        regrets = []
        for x_test, _, _ in test_loader:
            scores, test_probs = self.calc_test_batch_regret(net, x_candidates, x_test)
            num_candidates, num_labels = scores.shape[:2]

            # Duplicate to fit model_scores
            test_probs = (
                test_probs.unsqueeze(0)
                .unsqueeze(0)
                .repeat(num_candidates, num_labels, 1, 1, 1)
            )

            # Calculate the model that maximizes log_probs_candidates + log_probs_test
            _, chosen_model_idxs = scores.max(axis=-1)

            # Best model prediction probability
            max_probs = torch.gather(
                test_probs, -1, chosen_model_idxs.unsqueeze(-1)
            ).squeeze()

            # Regret for each test sample
            regrets_i = max_probs.sum(axis=-1).log()
            regrets.append(regrets_i)
        regrets = torch.cat(regrets, -1)
        return regrets

    def iterate_candidate_loader(self, net, candidate_loader, test_loader):
        net.train()
        regrets, candidate_idxs = [], []
        for x_candidates, _, candidate_idxs_b in tqdm(candidate_loader, desc="Query"):
            batch_regrets = self.iterate_test_loader(net, test_loader, x_candidates)
            regrets.append(batch_regrets)
            candidate_idxs.append(candidate_idxs_b)
        candidate_idxs = torch.hstack(candidate_idxs)

        # Average regret over the test set
        regrets = torch.cat(regrets, -1).mean(axis=-1)
        return regrets, candidate_idxs

    def regert_best_selection(self, mean_regrets, idx_candidates, dataset, n):
        max_y_regret, max_y_idx = mean_regrets.max(axis=-1)
        min_x_regret, min_x_idx = max_y_regret.sort(descending=False, axis=-1)

        wandb.log(
            {
                "regret_of_chosen": min_x_regret[0].item(),
                "max_yn_value": max_y_idx[0].item(),
                "max_yn_mean_value": max_y_idx.float().mean().item(),
            }
        )

        return idx_candidates[min_x_idx[:n]]

    def query(self, n, net, dataset):
        torch.set_grad_enabled(False)

        # Datasets
        candidate_loader, val_loader, test_loader = self.build_dataloaders(dataset)

        # self.temperature = net.calibrate(val_loader)
        # wandb.log({"temperature": self.temperature})
        # logger.info(f"Temperature: {self.temperature}")

        # Inference
        regrets, candidate_idxs = self.iterate_candidate_loader(
            net, candidate_loader, test_loader
        )

        choesn_idxs = self.regert_best_selection(regrets, candidate_idxs, dataset, n)
        torch.set_grad_enabled(True)

        # For debug:
        unlabled_labels = dataset.Y_train
        max_y_regret, _ = regrets.max(axis=-1)
        min_x_regret, min_x_idx = max_y_regret.sort(descending=False, axis=-1)
        logger.info(unlabled_labels[min_x_idx])
        logger.info(min_x_regret)
        return choesn_idxs
