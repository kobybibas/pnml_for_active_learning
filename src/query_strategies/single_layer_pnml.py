from .strategy import Strategy
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SingleLayerPnml(Strategy):
    def __init__(self, dataset, net):
        self.pinv_rcond = 1e-15
        super(SingleLayerPnml, self).__init__(dataset, net)

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

    def query(self, n):
        # Training set
        _, train_data = self.dataset.get_labeled_data()
        train_embeddings = self.get_embeddings(train_data)

        # Unlabeled
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_embeddings = self.get_embeddings(unlabeled_data)

        # Test
        test_data = self.dataset.get_test_data()
        test_probs = self.predict_prob(test_data)
        test_embeddings = self.get_embeddings(test_data)

        # Calc normalization factor
        nfs = []
        for unlabeled_vec in tqdm(unlabeled_embeddings):
            XN = torch.vstack((train_embeddings, unlabeled_vec))
            x_t_x = torch.matmul(XN.t(), XN)
            x_t_x_inv = torch.linalg.pinv(x_t_x, hermitian=False, rcond=self.pinv_rcond)
            nf = self.calc_norm_factor(x_t_x_inv, test_embeddings, test_probs).mean()
            nfs.append(nf)
        nfs = torch.hstack(nfs)
        nf_sorted, indices = torch.sort(nfs, descending=True)
        logger.info(nf_sorted)
        return unlabeled_idxs[indices[:n]]
