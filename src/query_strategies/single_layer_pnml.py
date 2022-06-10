import logging
import time
from typing import Tuple

import torch
import torch.linalg as tl
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .strategy import Strategy

logger = logging.getLogger(__name__)


class SingleLayerPnml(Strategy):
    def __init__(self, dataset, net):
        super(SingleLayerPnml, self).__init__(dataset, net)
        self.eps = 1e-3
        self.unlabeled_batch_size = 128
        self.unlabeled_pool_size = 4096

    def add_ones(self, embs: torch.Tensor) -> torch.Tensor:
        num_samples, feature_size = embs.shape
        return torch.hstack(
            [embs, torch.ones(num_samples, 1, device=embs.device, dtype=embs.dtype)]
        )

    def calc_P_N_minus_1(self, train_embs: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the traninings set correlation matrix
        """
        # Training set inverse:  num_features x num_features
        num_samples, num_features = train_embs.shape
        failed, num_iter, eps = True, 0, self.eps
        while failed:
            try:
                P_N_minus_1 = tl.inv(
                    train_embs.T.float() @ train_embs.float()
                    + eps * torch.eye(num_features).to(train_embs.device)
                ).half()
                if (
                    torch.isinf(P_N_minus_1).any().item()
                    or torch.isnan(P_N_minus_1).any().item()
                ):
                    raise Exception("isinf(P_N_minus_1)")
                failed = False
            except:
                num_iter += 1
                eps = min(eps * 10, 1)
                logger.info(f"P_N_minus_1 {num_iter} {eps=}. Failed")
        logger.info(
            f"Finished P_N_minus_1 {num_iter=} {eps=}. {[num_samples, num_features]=}"
        )
        return P_N_minus_1

    def calc_mul_vec_matrix_vec(
        self, vecs: torch.Tensor, mat: torch.Tensor
    ) -> torch.Tensor:
        # TODO: torch.einsum?
        return torch.hstack(
            [vec_i.unsqueeze(0) @ mat @ vec_i.unsqueeze(1) for vec_i in vecs]
        ).squeeze()

    def calc_x_P_N_minus_1_xt(
        self, test_embs: torch.Tensor, P_N_minus_1: torch.Tensor
    ) -> torch.Tensor:
        x_P_N_minus_1_xt = self.calc_mul_vec_matrix_vec(test_embs, P_N_minus_1)
        assert not torch.isnan(x_P_N_minus_1_xt).any().item()
        assert not torch.isinf(x_P_N_minus_1_xt).any().item()
        return x_P_N_minus_1_xt

    def calc_theta_c_n(
        self,
        xn: torch.Tensor,
        logit_n: torch.Tensor,
        theta_n_minus_1: torch.Tensor,
        P_N_minus_1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the values of the new paramters, taking into acount the unlabeled vector in the training set
        ------------------
        xn: unlabeled data
        theta_n_minus_1: learner based on the traiing set only
        P_N_minus_1: The inverse of the training set correlation matrix
        """
        num_labels = logit_n.size(-1)
        num_unlabeled, num_features = xn.shape

        # Eq 12 calc g: The projection of xn on the inv training correlation matrix. [num_unlabeled x 1]
        xn_P_N_minus_1_xnt = self.calc_mul_vec_matrix_vec(xn, P_N_minus_1).unsqueeze(-1)
        g = xn @ P_N_minus_1 / (1 + xn_P_N_minus_1_xnt)

        # Eq 16 z=f^{-1}(yn)=ln(S). [num_unlabeled x num_labels]
        z = torch.log(torch.sum(torch.exp(logit_n), axis=1, keepdims=True)).repeat(
            [1, num_labels]
        )

        # Eq. 12. [num_unlabeled x num_labels x num_features]:
        #     theta_c_n[n,m,l]: for unlabeled sample n added to the training set with the label m,
        #                       what would be the learner weight in the entry l
        theta_c_n = theta_n_minus_1.unsqueeze(0) + torch.bmm(
            g.unsqueeze(-1), (z - logit_n).unsqueeze(1)
        ).transpose(1, -1)

        assert theta_c_n.size(0) == num_unlabeled
        assert theta_c_n.size(1) == num_labels
        assert theta_c_n.size(2) == num_features
        assert not torch.isnan(theta_c_n).any().item()
        assert not torch.isinf(theta_c_n).any().item()
        return theta_c_n, xn_P_N_minus_1_xnt

    def calc_new_logits(
        self, logit_test_n_plus_1_for_c: torch.Tensor, test_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the new logits for the test set
        logit_test_n_plus_1_for_c: The new logit 'c' for the case an unlabeled sample was added to the training set with label 'c'
        test_logits: calculated based on the training set only. [test_size x num_labels]
        """
        unlabeled_size, num_labels, test_size = logit_test_n_plus_1_for_c.shape

        # Expend old test logits to match unlabeled size. [num_unlabeled x num_test x num_labels x num_labels]
        test_probs_n_plus_1 = (
            test_logits.unsqueeze(0)
            .unsqueeze(-2)
            .repeat([unlabeled_size, 1, num_labels, 1])
        )

        # Mask to insert new logits
        eye = (
            torch.eye(num_labels)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat([unlabeled_size, test_size, 1, 1])
            .bool()
        )

        # Inserting new logits
        test_probs_n_plus_1[eye] = logit_test_n_plus_1_for_c.transpose(1, -1).flatten()

        # Calculate test prediction with new logits: if we would add the unlabeled data to the training set
        test_probs_n_plus_1 = torch.nn.functional.softmax(test_probs_n_plus_1, dim=-1)

        assert test_probs_n_plus_1.size(0) == unlabeled_size
        assert test_probs_n_plus_1.size(1) == test_size
        assert test_probs_n_plus_1.size(2) == num_labels
        assert test_probs_n_plus_1.size(3) == num_labels
        assert not torch.isnan(test_probs_n_plus_1).any().item()
        assert not torch.isinf(test_probs_n_plus_1).any().item()
        return test_probs_n_plus_1

    def calc_test_emb_proj(
        self,
        test_embs: torch.Tensor,
        xn: torch.Tensor,
        P_N_minus_1: torch.Tensor,
        xn_P_N_minus_1_xnt: torch.Tensor,
        x_P_N_minus_1_xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the projection of the test embeddings on the inverse of the training+unlabeled correlation matrix
        test_embs: test set embeddings
        xn:  unlabeled set embeddings
        P_N_minus_1: The inverse of the training set correlation matrix
        xn_P_N_minus_1_xnt: The projection of the unlabeled embeddings on P_N_minus_1
        x_P_N_minus_1_xt: The projection of the test embeddings on P_N_minus_1
        """
        unlabeled_size, test_size = xn_P_N_minus_1_xnt.size(0), test_embs.size(0)

        a = xn_P_N_minus_1_xnt
        xn_P_N_minus_1_xt = torch.hstack(
            [test_embs @ P_N_minus_1 @ xn_i.unsqueeze(1) for xn_i in xn]
        ).transpose(1, 0)
        x_proj = torch.add(
            ((1 + a) / (1 + 2 * a)) @ x_P_N_minus_1_xt.unsqueeze(0),
            (-1 / (1 + 2 * a)) * (xn_P_N_minus_1_xt ** 2),
        )

        assert x_proj.size(0) == unlabeled_size
        assert x_proj.size(1) == test_size
        assert not torch.isnan(x_proj).any().item()
        assert not torch.isinf(x_proj).any().item()
        assert torch.all(x_proj >= 0)
        return x_proj

    def calc_normalization_factor(
        self, probs: torch.Tensor, x_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        
        probs: test_probs_n_plus_1
        """
        unlabeled_size, test_size, num_labeles, _ = probs.shape
        x_proj = x_proj.unsqueeze(-1).unsqueeze(-1)

        # Eq. 20
        nf = torch.sum(probs / (probs + (1 - probs) * (probs ** x_proj)), dim=-1)
        assert nf.size(0) == unlabeled_size
        assert nf.size(1) == test_size
        assert nf.size(2) == num_labeles
        assert torch.all(nf > 1)
        assert not torch.isnan(nf).any().item()
        assert not torch.isinf(nf).any().item()
        return nf

    def query(self, n):
        logger.info("SingleLayerPnml: query")
        torch.set_grad_enabled(False)

        # Training set
        t1 = time.time()
        _, train_data = self.dataset.get_labeled_data()
        train_embs = self.get_embeddings(train_data)
        train_embs = self.add_ones(train_embs) # .float()
        training_size, num_features = train_embs.shape
        logger.info(
            f"Training: get_embeddings in {time.time()-t1:.1f}. {[training_size, num_features]=}"
        )

        # Unlabeled set
        t1 = time.time()
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_embs, unlabeled_logits, _ = self.get_emb_logit_prob(unlabeled_data)
        # unlabeled_embs, unlabeled_logits = (
        #     unlabeled_embs.float(),
        #     unlabeled_logits.float(),
        # )
        # -- Reduce number of unlabeled to evaluate:
        idxs =  torch.randperm(len(unlabeled_embs))[:self.unlabeled_pool_size]
        unlabeled_embs, unlabeled_logits, unlabeled_idxs = (
            unlabeled_embs[idxs],
            unlabeled_logits[idxs],
            unlabeled_idxs[idxs],
        )
        unlabeled_embs = self.add_ones(unlabeled_embs)
        unlabeled_size, num_features = unlabeled_embs.shape
        logger.info(
            f"Unlabeled: get_embeddings in {time.time()-t1:.1f}. {[unlabeled_size, num_features]=}"
        )

        # Test set
        t1 = time.time()
        test_embs, test_logits, _ = self.get_emb_logit_prob(
            self.dataset.get_test_data()
        )
        # test_embs, test_logits = test_embs.float(), test_logits.float()
        test_embs = self.add_ones(test_embs)
        test_size, num_features = test_embs.shape
        logger.info(
            f"Test: get_embeddings in {time.time()-t1:.1f}. {[test_size, num_features]=}"
        )

        # Inverse of training set
        P_N_minus_1 = self.calc_P_N_minus_1(train_embs)

        # ERM classifier: num_features x num_labels
        clf = self.net.clf.get_classifer()
        theta_n_minus_1 = (
            torch.hstack([clf.weight, clf.bias.unsqueeze(-1)]).clone() # .float()
        )

        # Dataloder
        unlabeled_dataloader = DataLoader(
            TensorDataset(unlabeled_embs, unlabeled_logits),
            num_workers=0,
            batch_size=self.unlabeled_batch_size,
        )

        # Test projection on training design matrix
        t1 = time.time()
        x_P_N_minus_1_xt = self.calc_x_P_N_minus_1_xt(test_embs, P_N_minus_1)
        logger.info(
            f"calc_x_P_N_minus_1_xt in {time.time()-t1:.1f} sec. {x_P_N_minus_1_xt.shape=}"
        )

        max_yn_for_nf_list = []
        for xn, logit_n in tqdm(unlabeled_dataloader):
            # xn: num_unlabeled_batch x num_features
            # logit_n: num_unlabeled_batch x num_labels. xn_t @ theta_n_minus_1
            unlabeled_size, _ = xn.shape

            # Calculate the new learner with the unlabeled data
            theta_c_n, xn_P_N_minus_1_xnt = self.calc_theta_c_n(
                xn, logit_n, theta_n_minus_1, P_N_minus_1
            )

            # The new logits with the unlabeled data in the training set. the diagonal of test_logits_expanded
            # [unlabeled_size x num_labels x test_size]
            logit_test_n_plus_1_for_c = theta_c_n @ test_embs.transpose(0, -1)

            # New probability assignmnet. [unlabeled_size x test_size x num_labels x num_labels]
            test_probs_n_plus_1 = self.calc_new_logits(
                logit_test_n_plus_1_for_c, test_logits
            )

            # Test projection on the design matrix with unlabled
            x_proj = self.calc_test_emb_proj(
                test_embs, xn, P_N_minus_1, xn_P_N_minus_1_xnt, x_P_N_minus_1_xt
            )

            # Normalization factor
            nf = self.calc_normalization_factor(test_probs_n_plus_1, x_proj)

            # worst y_n: Average y_n of xn over al the training set, then take the worst
            max_yn_for_nf = nf.mean(axis=1).max(dim=-1)[0]

            max_yn_for_nf_list.append(max_yn_for_nf)
        max_yn_for_nf = torch.hstack(max_yn_for_nf_list)

        # Sort from min to max
        min_xn_values, min_xn_idx = max_yn_for_nf.sort(descending=False)
        logger.info(
            f"{min_xn_values[:2]=}, {min_xn_values[-2:]=}, {min_xn_values.mean().item()=}"
        )

        torch.set_grad_enabled(True)
        return unlabeled_idxs[min_xn_idx[:n]]
