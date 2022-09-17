import logging
import time
from time import time
from typing import Tuple

import torch
import torch.linalg as tl
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from .strategy import Strategy

logger = logging.getLogger(__name__)


class SingleLayerPnml(Strategy):
    def __init__(self):
        self.eps = 1e-6
        self.unlabeled_batch_size = 64
        self.unlabeled_pool_size = 1000
        self.test_set_size = 1000
        self.parameter_lr = 1.0

    def add_bias_term(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hstack([x, torch.ones(len(x), 1, device=x.device, dtype=x.dtype)])

    def calc_PN_minus_1(self, train_embs: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the traninings set correlation matrix
        """
        cov = train_embs.T @ train_embs
        try:
            PN_minus_1 = tl.inv(cov)
        except:
            PN_minus_1 = tl.inv(cov)
        assert not torch.isinf(PN_minus_1).any().item()
        assert not torch.isnan(PN_minus_1).any().item()
        return PN_minus_1

    def calc_mul_vec_matrix_vec(
        self, vecs: torch.Tensor, mat: torch.Tensor
    ) -> torch.Tensor:
        results = torch.diag(vecs @ mat @ vecs.transpose(1, 0))
        assert not torch.isnan(results).any().item()
        assert not torch.isinf(results).any().item()
        assert (results >= 0).all().item()
        return results

    def calc_theta_c_n(
        self,
        xn: torch.Tensor,
        logit_n: torch.Tensor,
        prob_n: torch.Tensor,
        theta_n_minus_1: torch.Tensor,
        PN_minus_1: torch.Tensor,
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
        xn_PN_minus_1_xnt = self.calc_mul_vec_matrix_vec(xn, PN_minus_1).unsqueeze(-1)
        g = xn @ PN_minus_1 / (1 + xn_PN_minus_1_xnt)

        # Eq 16 z=f^{-1}(yn)=ln(S). [num_unlabeled x num_labels]
        z = torch.logsumexp(logit_n, axis=1, keepdims=True).repeat([1, num_labels])
        diff = z - logit_n
        assert (diff >= 0).all().item()

        # Eq. 12. [num_unlabeled x num_labels x num_features]:
        #     theta_c_n[n,m,l]: for unlabeled sample n added to the training set with the label m,
        #                       what would be the learner weight in the entry l
        theta_c_n = theta_n_minus_1.unsqueeze(0) + self.parameter_lr * torch.bmm(
            g.unsqueeze(-1), diff.unsqueeze(1)
        ).transpose(1, -1)

        assert theta_c_n.size(0) == num_unlabeled
        assert theta_c_n.size(1) == num_labels
        assert theta_c_n.size(2) == num_features
        assert not torch.isnan(theta_c_n).any().item()
        assert not torch.isinf(theta_c_n).any().item()
        return theta_c_n, xn_PN_minus_1_xnt

    def calc_new_logits(
        self, logit_test_n_for_c: torch.Tensor, test_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the new logits for the test set
        logit_test_n_for_c: The new logit 'c' for the case an unlabeled sample was added to the training set with label 'c'
        test_logits: calculated based on the training set only. [test_size x num_labels]
        """
        unlabeled_size, num_labels, test_size = logit_test_n_for_c.shape

        # Expend old test logits to match unlabeled size. [num_unlabeled x num_test x num_labels x num_labels]
        test_probs_n = (
            test_logits.unsqueeze(0)
            .unsqueeze(0)
            .repeat([unlabeled_size, num_labels, 1, 1])
        )

        # Mask to insert new logits
        eye = (
            torch.eye(num_labels)
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat([unlabeled_size, 1, test_size, 1])
            .bool()
        )

        # Inserting new logits
        test_probs_n[eye] = logit_test_n_for_c.flatten()

        # Calculate test prediction with new logits: if we would add the unlabeled data to the training set
        test_probs_n = torch.nn.functional.softmax(test_probs_n, dim=-1)

        assert test_probs_n.size(0) == unlabeled_size
        assert test_probs_n.size(1) == num_labels
        assert test_probs_n.size(2) == test_size
        assert test_probs_n.size(3) == num_labels
        assert not torch.isnan(test_probs_n).any().item()
        assert not torch.isinf(test_probs_n).any().item()
        return test_probs_n

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
        unlabeled_size, num_labeles, test_size, _ = probs.shape

        # Eq. 20
        nf = torch.sum(
            probs
            / (
                probs
                + (1 - probs) * (probs ** x_proj.unsqueeze(-1).unsqueeze(1))
                + self.eps
            ),
            dim=-1,
        )
        assert nf.size(0) == unlabeled_size
        assert nf.size(1) == num_labeles
        assert nf.size(2) == test_size
        assert torch.all(nf >= (1 - 1e-3)), print(nf[nf < 1])
        assert not torch.isnan(nf).any().item()
        assert not torch.isinf(nf).any().item()
        return nf

    def choose_single_unlabel(
        self, unlabeled_dataloader, theta_n_minus_1, train_embs, test_embs, test_logits,
    ):

        # Inverse of training set
        PN_minus_1 = self.calc_PN_minus_1(train_embs)

        # Test projection on training design matrix
        x_PN_minus_1_xt = self.calc_mul_vec_matrix_vec(test_embs, PN_minus_1)

        nfs_with_unlabeled, theta_c_ns, true_labels = [], [], []
        for unlabeled_xn, unlabeled_logits, unlabeled_probs, unlabeled_labels in tqdm(
            unlabeled_dataloader
        ):

            # Calculate the new learner with the unlabeled data
            theta_c_n, xn_PN_minus_1_xnt = self.calc_theta_c_n(
                unlabeled_xn,
                unlabeled_logits,
                unlabeled_probs,
                theta_n_minus_1,
                PN_minus_1,
            )

            # The new logits with the unlabeled data in the training set. the diagonal of test_logits_expanded
            # [unlabeled_size x num_labels x test_size]
            logit_test_n_for_c = theta_c_n @ test_embs.transpose(0, -1)

            # New probability assignmnet. [unlabeled_size x test_size x num_labels x num_labels]
            test_probs_n = self.calc_new_logits(logit_test_n_for_c, test_logits)

            # Test projection on the design matrix with unlabled
            x_proj = self.calc_test_emb_proj(
                test_embs, unlabeled_xn, PN_minus_1, xn_PN_minus_1_xnt, x_PN_minus_1_xt
            )

            # Normalization factor. [nulabeled_size x num_labels x test_size]
            nf = self.calc_normalization_factor(test_probs_n, x_proj)

            # Save: Average y_n over the test set,
            nfs_with_unlabeled.append(nf.mean(axis=-1))
            theta_c_ns.append(theta_c_n)
            true_labels.append(unlabeled_labels)

        nfs_with_unlabeled = torch.vstack(nfs_with_unlabeled)
        theta_c_ns = torch.vstack(theta_c_ns)

        # Worst y_n: Average y_n the test set, then take the worst y_n
        nf_of_max_yn, max_yn_values = nfs_with_unlabeled.max(dim=-1)
        min_xn_values, min_xn_indices = nf_of_max_yn.sort()
        min_xn_idx = min_xn_indices[0]
        if False:
            # Pick randomly from the top 10 candidates
            min_xn_idx = min_xn_indices[torch.randint(0, 10, (1,))]
        max_yn_value = max_yn_values[min_xn_idx]
        nf_of_chosen = nf_of_max_yn[min_xn_idx]

        if False:
            # Genie that knows the true unlabled label
            true_labels = torch.hstack(true_labels)
            nf_of_chosen, min_xn_idx = torch.hstack(
                [
                    nfs_with_unlabeled[i, true_label]
                    for i, true_label in enumerate(true_labels)
                ]
            ).min(dim=0)
        if False:
            # Get label of the least frequent in the training set
            max_yn_value = self.label_to_get
            min_xn_idx = nfs_with_unlabeled[:, self.label_to_get].argmin()

        # Classifer than was trained with xn and max_yn
        theta_c_n = theta_c_ns[min_xn_idx, max_yn_value]

        # Normalization factor without unlabeled
        test_probs = torch.softmax(test_logits, axis=-1).unsqueeze(0).unsqueeze(0)
        x_proj_on_trainset = x_PN_minus_1_xt.unsqueeze(0)
        nf_current = self.calc_normalization_factor(
            test_probs, x_proj_on_trainset
        ).mean()

        return (
            min_xn_idx.item(),
            max_yn_value.item(),
            theta_c_n,
            nf_of_chosen.item(),
            nf_current,
        )

    def query(self, n, net, dataset):
        torch.set_grad_enabled(False)

        # Training set
        t1 = time()
        train_labels = dataset.get_labeled_labels()
        self.label_to_get = train_labels.bincount().argmin()
        train_embs, _, _ = net.get_emb_logit_prob(dataset.get_labeled_data()[1])
        train_embs = self.add_bias_term(train_embs)
        logger.info(f"get_embeddings in {time()-t1:.1f}. {train_embs.shape=}")

        # Unlabeled set
        t1 = time()
        unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
        unlabeled_embs, unlabeled_logits, unlabeled_probs = net.get_emb_logit_prob(
            unlabeled_data
        )
        idxs = torch.randperm(len(unlabeled_embs))[: self.unlabeled_pool_size]
        unlabeled_embs = unlabeled_embs[idxs]
        unlabeled_logits = unlabeled_logits[idxs]
        unlabeled_probs = unlabeled_probs[idxs]
        unlabeled_idxs = unlabeled_idxs[idxs]  # Used to debug
        unlabeled_labels = dataset.Y_train[unlabeled_idxs]
        unlabeled_embs = self.add_bias_term(unlabeled_embs)
        logger.info(f"get_embeddings in {time()-t1:.1f}. {unlabeled_embs.shape=}")

        # Test set
        t1 = time()
        test_embs, test_logits, _ = net.get_emb_logit_prob(dataset.get_test_data())
        idxs = torch.randperm(len(test_embs))[: self.test_set_size]
        test_embs = test_embs[idxs]
        test_logits = test_logits[idxs]
        test_embs = self.add_bias_term(test_embs)
        logger.info(f"get_embeddings in {time()-t1:.1f}. {test_embs.shape=}")

        # ERM classifier: num_features x num_labels
        clf = net.clf.get_classifer()
        theta_n_minus_1 = torch.hstack([clf.weight, clf.bias.unsqueeze(-1)]).clone()

        chosen_idxs = []
        for n_i in range(n):

            unlabeled_dataloader = DataLoader(
                TensorDataset(
                    unlabeled_embs, unlabeled_logits, unlabeled_probs, unlabeled_labels
                ),
                num_workers=0,
                batch_size=self.unlabeled_batch_size,
                shuffle=False,
            )

            (
                min_xn_idx,
                max_yn_value,
                theta_c_n,
                nf_of_chosen,
                nf_current,
            ) = self.choose_single_unlabel(
                unlabeled_dataloader,
                theta_n_minus_1,
                train_embs,
                test_embs,
                test_logits,
            )

            # Save chosen index
            chosen_unlabeled_idx = unlabeled_idxs[min_xn_idx]
            chosen_unlabeled_emb = unlabeled_embs[min_xn_idx]
            chosen_idxs.append(chosen_unlabeled_idx)

            # Remove chosen index from unlabel
            unlabeled_embs = torch.vstack(
                (unlabeled_embs[:min_xn_idx], unlabeled_embs[min_xn_idx + 1 :])
            )
            unlabeled_logits = torch.vstack(
                (unlabeled_logits[:min_xn_idx], unlabeled_logits[min_xn_idx + 1 :])
            )
            unlabeled_probs = torch.vstack(
                (unlabeled_probs[:min_xn_idx], unlabeled_probs[min_xn_idx + 1 :])
            )

            # Define new training set
            train_embs = torch.vstack((train_embs, chosen_unlabeled_emb))
            theta_n_minus_1 = theta_c_n

            # Logging
            actual_yn_value = dataset.Y_train[chosen_unlabeled_idx]
            logger.info(
                f"[{n_i}/{n-1}] yn_value: [max actual]=[{max_yn_value} {actual_yn_value}]. {nf_of_chosen=}"
            )

            wandb.log(
                {
                    "nf_of_chosen": nf_of_chosen,
                    "actual_yn_value": actual_yn_value,
                    "max_yn_value": max_yn_value,
                    "nf_current": nf_current,
                }
            )
        torch.set_grad_enabled(True)
        return chosen_idxs

