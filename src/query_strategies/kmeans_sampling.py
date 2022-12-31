import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class KMeansSampling(Strategy):
    def __init__(self):
        super(KMeansSampling, self).__init__()

    def query(self, n, net, dataset):
        logger.info("KMeansSampling")
        unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
        embeddings = net.get_embeddings(unlabeled_data)
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
