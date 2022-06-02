from .strategy import Strategy
import torch
from tqdm import tqdm
import logging

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import logging
import time
from torch.utils.data import DataLoader
import torch.linalg as tl

logger = logging.getLogger(__name__)


class RegretCalculator(Dataset):
    def __init__(
        self, train_embeddings, unlabeled_embeddings, test_embeddings, test_probs
    ):
        self.train_embeddings = train_embeddings.float().cpu()
        self.unlabeled_embeddings = unlabeled_embeddings.float().cpu()
        self.test_embeddings = test_embeddings.float().cpu()
        self.test_probs = test_probs.float().cpu()
        self.pinv_rcond = 1e-15

    def __getitem__(self, index):
        unlabeled_vec = self.unlabeled_embeddings[index]

        XN = torch.vstack((self.train_embeddings, unlabeled_vec))
        x_t_x = torch.matmul(XN.t(), XN)
        x_t_x_inv = tl.pinv(x_t_x, hermitian=False, rcond=self.pinv_rcond)
        nf = self.calc_norm_factor(
            x_t_x_inv, self.test_embeddings, self.test_probs
        ).mean()
        return nf, index

    def add_bias_term(self, embeddings, bias):
        torch.hstack(embeddings, torch.ones)

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

    def __len__(self):
        return len(self.unlabeled_embeddings)


class SingleLayerPnml(Strategy):
    def __init__(self, dataset, net):
        super(SingleLayerPnml, self).__init__(dataset, net)

    def query(self, n):
        torch.set_grad_enabled(False)
        # Training set
        _, train_data = self.dataset.get_labeled_data()
        train_embeddings = self.get_embeddings(train_data)

        # Unlabeled
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_embeddings, unlabled_probs = self.get_embddings_and_prob(
            unlabeled_data
        )

        # Test
        test_data = self.dataset.get_test_data()
        test_embeddings, test_probs = self.get_embddings_and_prob(test_data)

        if False:
            # Calc normalization factor
            regret_calc_h = RegretCalculator(
                train_embeddings, unlabeled_embeddings, test_embeddings, test_probs
            )
            loader = DataLoader(
                regret_calc_h, shuffle=False, batch_size=128, num_workers=4
            )

            nfs = torch.zeros(len(regret_calc_h))
            for nf, idx in tqdm(loader):
                nfs[idx] = nf
            nf_sorted, indices = torch.sort(nfs, descending=True)
            logger.info(nf_sorted)
            return unlabeled_idxs[indices[:n]]

        xt_x_inv = tl.pinv(
            train_embeddings.float().T @ train_embeddings.float()
        ).half()  # num_features x num_features

        # Classifier with the unlabeled samples: num_unlabled x num_labels
        clf = self.net.clf.get_classifer()
        theta_n_minus_1 = clf.weight  # TODO: bias

        eps = torch.finfo(torch.float16).eps
        z = (
            torch.log(unlabled_probs + eps)
            + torch.log(
                torch.sum(torch.exp(clf(unlabeled_embeddings)), axis=1, keepdims=True)
            )
        ).unsqueeze(
            1
        )  # eq 16: num_unlabeled x 1 x num_features

        # Calculate g: eq 12.
        xnt_inv_xn = torch.hstack(
            [
                xn.unsqueeze(0) @ xt_x_inv @ xn.unsqueeze(1)
                for xn in unlabeled_embeddings
            ]
        ) # TODO:  torch.einsum?

        gt = xt_x_inv @ unlabeled_embeddings.T / (1 + xnt_inv_xn)
        g = torch.transpose(gt, 1, 0).unsqueeze(-1)  # num_unlabeled x num_features x 1

        #
        unlabeled_logits = clf(unlabeled_embeddings).unsqueeze(1) # xt @ theta_n_minus_1
        c = torch.vstack([(a @ b).unsqueeze(0) for a,b in  zip(g,unlabeled_logits )]).transpose(2,1) # num_unlabled x num_features x num_labeles
        theta_n_minus_1 = c
        # theta_n = theta_n_minus_1 + torch.bmm(g , 
        #     z - clf(unlabeled_embeddings)
        # )  #  eq 12. num_unlabled x num_labels

        logit_test_n_plus_1 =  theta_n_minus_1 @ test_embeddings.transpose(1,0) # num_unlabeled x num_test x num_labels
        logit_test_n_plus_1 = logit_test_n_plus_1.transpose(1,2)

        # Calc inv of each train with unlabeled: with matrix lemma. num_unlabled x num_features x num_fearues
        batch_size = 128
        unlabeled_embeddings_b = unlabeled_embeddings[:batch_size]
        xnt_inv_xn_b = xnt_inv_xn[:,:batch_size]
        xnt_inv_xn_b = xnt_inv_xn_b.squeeze().unsqueeze(-1).unsqueeze(-1)

        a0 = torch.vstack([(xt_x_inv @ xn.unsqueeze(1) @ xn.unsqueeze(0) @ xt_x_inv).unsqueeze(0)  for xn in unlabeled_embeddings_b])
        xt_x_inv_n = xt_x_inv.unsqueeze(0) - a0/xnt_inv_xn_b

        regres = None  # eq 20. num_unlabled x num_labels
