from .strategy import Strategy
import torch
from tqdm import tqdm
import logging

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import logging
import time
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class RegretCalculator(Dataset):
    def __init__(
        self, train_embeddings, unlabeled_embeddings, test_embeddings, test_probs
    ):
        self.train_embeddings = train_embeddings.float().cpu()
        self.unlabeled_embeddings = unlabeled_embeddings.float().cpu()
        self.test_embeddings = test_embeddings.float().cpu()
        self.test_probs = test_probs.float().cpu()
        self.pinv_rcond = 1e-15

    def __getitem__(self, index):
        unlabeled_vec = self.unlabeled_embeddings[index]

        XN = torch.vstack((self.train_embeddings, unlabeled_vec))
        x_t_x = torch.matmul(XN.t(), XN)
        x_t_x_inv = torch.linalg.pinv(x_t_x, hermitian=False, rcond=self.pinv_rcond)
        nf = self.calc_norm_factor(
            x_t_x_inv, self.test_embeddings, self.test_probs
        ).mean()
        return nf, index

    def calc_norm_factor(
        self, x_t_x_inv, test_features: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        device = test_features.device

        x_proj = torch.abs(
            torch.matmul(
                torch.matmul(test_features.unsqueeze(1), x_t_x_inv),
                test_features.unsqueeze(-1),
            ).squeeze(-1)
        )
        x_t_g = x_proj / (1 + x_proj)

        # Compute the normalization factor
        probs = probs.to(device)
        n_classes = probs.shape[-1]
        nf = torch.sum(probs / (probs + (1 - probs) * (probs ** x_t_g)), dim=-1)
        return nf

    def __len__(self):
        return len(self.unlabeled_embeddings)


class SingleLayerPnml(Strategy):
    def __init__(self, dataset, net):
        super(SingleLayerPnml, self).__init__(dataset, net)

    def query(self, n):
        # Training set
        _, train_data = self.dataset.get_labeled_data()
        train_embeddings = self.get_embeddings(train_data).float()

        # Unlabeled
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_embeddings = self.get_embeddings(unlabeled_data).float()

        # Test
        test_data = self.dataset.get_test_data()
        test_probs = self.predict_prob(test_data)
        test_embeddings = self.get_embeddings(test_data).float()

        # Calc normalization factor
        regret_calc_h = RegretCalculator(
            train_embeddings, unlabeled_embeddings, test_embeddings, test_probs
        )
        loader = DataLoader(regret_calc_h, shuffle=False, batch_size=128, num_workers=4)

        nfs = torch.zeros(len(regret_calc_h))
        for nf, idx in tqdm(loader):
            nfs[idx] = nf
        nf_sorted, indices = torch.sort(nfs, descending=True)
        logger.info(nf_sorted)
        return unlabeled_idxs[indices[:n]]
