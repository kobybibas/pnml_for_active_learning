import logging

import numpy as np
from tqdm import tqdm

from .strategy import Strategy

logger = logging.getLogger(__name__)


class KCenterGreedy(Strategy):
    def query(self, n, net, dataset):
        logger.info("KCenterGreedy")
        labeled_idxs, train_data = dataset.get_train_data()
        embeddings = net.get_embeddings(train_data)
        embeddings = embeddings.cpu().numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
        return np.arange(dataset.n_pool)[(dataset.labeled_idxs ^ labeled_idxs)]
