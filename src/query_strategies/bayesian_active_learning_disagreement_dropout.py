import torch
from .strategy import Strategy
from torch.utils.data import Subset


class BALDDropout(Strategy):
    def __init__(
        self,
        n_drop: int = 10,
        unlabeled_pool_size: int = 256,
    ):
        self.n_drop = n_drop
        self.unlabeled_pool_size = unlabeled_pool_size

    def query(self, n, net, dataset):
        unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()

        idx_subset = torch.randperm(len(unlabeled_idxs))[: self.unlabeled_pool_size]
        unlabeled_idxs = unlabeled_idxs[idx_subset]
        candidate_subset = Subset(unlabeled_data, idx_subset)

        probs = net.predict_prob_dropout_split(candidate_subset, n_drop=self.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb * torch.log(pb)).sum(1)
        entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
