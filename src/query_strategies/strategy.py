from typing import Tuple

import torch
from data import Data
from nets import Net
from torch.utils.data import DataLoader


class Strategy:
    def __init__(self, dataset: Data, net: Net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data: Data, n_drop: int = 10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data: Data, n_drop: int = 10) -> torch.Tensor:
        return self.net.predict_prob_dropout_split(data, n_drop=n_drop)

    def get_embeddings(self, data: Data) -> torch.Tensor:
        return self.net.get_embeddings(data).float()

    def get_embddings_and_prob(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net.get_embddings_and_prob(data)
