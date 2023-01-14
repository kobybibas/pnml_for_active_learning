import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import logging
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class KMeansSampling(Strategy):
    def __init__(self, unlabeled_pool_size=-1):
        self.unlabeled_pool_size = unlabeled_pool_size
        super(KMeansSampling, self).__init__()

    def query(self, n, net, dataset):
        logger.info("KMeansSampling")
        unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
        # shuffle and constrain the size of the unlabeled pool
        candidate_idx_subset = np.random.permutation(len(unlabeled_idxs))[
            : self.unlabeled_pool_size
        ]
        unlabeled_idxs = unlabeled_idxs[candidate_idx_subset]
        unlabeled_subset = Subset(unlabeled_data, candidate_idx_subset)

        embeddings = net.get_embeddings(unlabeled_subset)
        embeddings = embeddings.cpu().numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)

        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [
                np.arange(embeddings.shape[0])[cluster_idxs == i][
                    dis[cluster_idxs == i].argmin()
                ]
                for i in range(n)
            ]
        )

        return unlabeled_idxs[q_idxs]
