import numpy as np


class Strategy:
    def query(self, n, net, dataset):
        pass

    def update(self, dataset, pos_idxs, neg_idxs=None):
        assert np.any(dataset.labeled_idxs[pos_idxs]) == False
        dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            dataset.labeled_idxs[neg_idxs] = False
