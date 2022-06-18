import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
    def query(self, n, net, dataset):
        return np.random.choice(
            np.where(dataset.labeled_idxs == 0)[0], n, replace=False
        )
