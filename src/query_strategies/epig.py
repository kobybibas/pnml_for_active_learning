"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

import logging
import math
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from subprocess import check_output
from typing import Callable, Sequence, Union
from torch.nn.functional import log_softmax, nll_loss
import numpy as np
import pandas as pd
import torch
import wandb
from numpy.random import Generator
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .strategy import Strategy

logger = logging.getLogger(__name__)


Array = Union[np.ndarray, Tensor]


class Dictionary(dict):
    def append(self, update: dict) -> None:
        for key in update:
            try:
                self[key].append(update[key])
            except:
                self[key] = [update[key]]

    def extend(self, update: dict) -> None:
        for key in update:
            try:
                self[key].extend(update[key])
            except:
                self[key] = update[key]

    def concatenate(self) -> dict:
        scores = deepcopy(self)
        for key in scores.keys():
            scores[key] = torch.cat(scores[key])
        return scores

    def numpy(self) -> dict:
        scores = deepcopy(self)
        for key in scores.keys():
            scores[key] = scores[key].numpy()
        return scores

    def subset(self, inds: Sequence) -> dict:
        scores = deepcopy(self)
        for key in scores.keys():
            scores[key] = scores[key][inds]
        return scores

    def save_to_csv(
        self, filepath: Path, formatting: Union[Callable, dict] = None
    ) -> None:
        table = pd.DataFrame(self)

        if callable(formatting):
            table = table.applymap(formatting)

        elif isinstance(formatting, dict):
            for key in formatting.keys():
                table[key] = table[key].apply(formatting[key])

        table.to_csv(filepath, index=False)

    def save_to_npz(self, filepath: Path) -> None:
        np.savez(filepath, **self)


def format_time(seconds: float) -> str:
    hours, minutes, seconds = str(timedelta(seconds=seconds)).split(":")
    return f"{int(hours):02}:{minutes}:{float(seconds):02.0f}"


def get_repo_status() -> dict:
    """
    References:
        https://stackoverflow.com/a/21901260
    """
    status = {
        "branch.txt": check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit.txt": check_output(["git", "rev-parse", "HEAD"]),
        "uncommitted.diff": check_output(["git", "diff"]),
    }
    return status


def set_rngs(seed: int = -1, constrain_cudnn: bool = False) -> Generator:
    """
    References:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed == -1:
        seed = random.randint(0, 1000)

    rng = np.random.default_rng(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if constrain_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return rng


def check(
    scores: Tensor,
    max_value: float = math.inf,
    epsilon: float = 1e-6,
    score_type: str = "",
) -> Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.
    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(
            f"Invalid {score_type} score (min = {min_score}, max = {max_score})"
        )

    return scores


def logmeanexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """
    Arguments:
        x: Tensor[float]
        dim: int
        keepdim: bool
    Returns:
        Tensor[float]
    """
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])


def marginal_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    See marginal_entropy_from_logprobs.
    Arguments:
        probs: Tensor[float], [N, K, Cl]
    Returns:
        Tensor[float], [N,]
    """
    probs = torch.mean(probs, dim=1)  # [N, Cl]
    scores = entropy_from_probs(probs)  # [N,]
    scores = check(scores, math.log(probs.shape[-1]), score_type="ME")  # [N,]
    return scores  # [N,]


def conditional_epig_from_logprobs(
    logprobs_pool: Tensor, logprobs_targ: Tensor
) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]
    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the log of the joint predictive distribution.
    logprobs_pool = logprobs_pool.permute(1, 0, 2)  # [K, N_p, Cl]
    logprobs_targ = logprobs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    logprobs_pool = logprobs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
    logprobs_targ = logprobs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
    logprobs_pool_targ_joint = logprobs_pool + logprobs_targ  # [K, N_p, N_t, Cl, Cl]
    logprobs_pool_targ_joint = logmeanexp(
        logprobs_pool_targ_joint, dim=0
    )  # [N_p, N_t, Cl, Cl]

    # Estimate the log of the marginal predictive distributions.
    logprobs_pool = logmeanexp(logprobs_pool, dim=0)  # [N_p, 1, Cl, 1]
    logprobs_targ = logmeanexp(logprobs_targ, dim=0)  # [1, N_t, 1, Cl]

    # Estimate the log of the product of the marginal predictive distributions.
    logprobs_pool_targ_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    probs_pool_targ_joint = torch.exp(logprobs_pool_targ_joint)  # [N_p, N_t, Cl, Cl]
    log_term = (
        logprobs_pool_targ_joint - logprobs_pool_targ_joint_indep
    )  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    return scores  # [N_p, N_t]


def conditional_epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    See conditional_epig_from_logprobs.
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the joint predictive distribution.
    probs_pool = probs_pool.permute(1, 0, 2)  # [K, N_p, Cl]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_pool = probs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
    probs_targ = probs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
    probs_pool_targ_joint = probs_pool * probs_targ  # [K, N_p, N_t, Cl, Cl]
    probs_pool_targ_joint = torch.mean(
        probs_pool_targ_joint, dim=0
    )  # [N_p, N_t, Cl, Cl]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=0)  # [N_p, 1, Cl, 1]
    probs_targ = torch.mean(probs_targ, dim=0)  # [1, N_t, 1, Cl]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    nonzero_joint = probs_pool_targ_joint > 0  # [N_p, N_t, Cl, Cl]
    log_term = torch.clone(probs_pool_targ_joint)  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] = torch.log(
        probs_pool_targ_joint[nonzero_joint]
    )  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] -= torch.log(
        probs_pool_targ_indep[nonzero_joint]
    )  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    return scores  # [N_p, N_t]


def epig_from_conditional_scores(scores: Tensor) -> Tensor:
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t]
    Returns:
        Tensor[float], [N_p,]
    """
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]


def epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]
    Returns:
        Tensor[float], [N_p,]
    """
    logger.info(
        f"Epig: epig_from_logprobs {logprobs_pool.shape=} {logprobs_targ.shape=}"
    )
    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


def epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    See epig_from_logprobs.
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


def epig_from_logprobs_using_matmul(
    logprobs_pool: Tensor, logprobs_targ: Tensor
) -> Tensor:
    """
    See epig_from_probs_using_matmul.
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]
    Returns:
        Tensor[float], [N_p,]
    """
    logger.info(
        f"Epig: epig_from_logprobs_using_matmul {logprobs_pool.shape=} {logprobs_targ.shape=}"
    )
    probs_pool = torch.exp(logprobs_pool)  # [N_p, K, Cl]
    probs_targ = torch.exp(logprobs_targ)  # [N_t, K, Cl]
    return epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]


def epig_from_probs_using_matmul(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x) = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = H[p(y|x)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]
    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).
    References:
        https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
    Returns:
        Tensor[float], [N_p,]
    """
    N_t, K, C = probs_targ.shape

    entropy_pool = marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = marginal_entropy_from_probs(probs_targ)  # [N_t,]
    entropy_targ = torch.mean(entropy_targ)  # [1,]

    probs_pool = probs_pool.permute(0, 2, 1)  # [N_p, Cl, K]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_targ = probs_targ.reshape(K, N_t * C)  # [K, N_t * Cl]
    probs_pool_targ_joint = probs_pool @ probs_targ / K  # [N_p, Cl, N_t * Cl]

    entropy_pool_targ = (
        -torch.sum(
            probs_pool_targ_joint * torch.log(probs_pool_targ_joint), dim=(-2, -1)
        )
        / N_t
    )  # [N_p,]
    entropy_pool_targ[torch.isnan(entropy_pool_targ)] = 0.0

    scores = entropy_pool + entropy_targ - entropy_pool_targ  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]


def epig_from_logprobs_using_weights(
    logprobs_pool: Tensor, logprobs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
            = ∫ p_*(x_*) EPIG(x|x_*) dx_*
            ~= ∫ p_{pool}(x_*) w(x_*) EPIG(x|x_*) dx_*
            ~= (1 / M) ∑_{i=1}^M w(x_*^i) EPIG(x|x_*^i)  where  x_*^i in D_{pool}
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl], preds on proxy target inputs from the pool
        weights: Tensor[float], [N_t,], weight on each proxy target input
    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


def epig_from_probs_using_weights(
    probs_pool: Tensor, probs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    See epig_from_logprobs_using_weights.
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        weights: Tensor[float], [N_t,]
    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


class Epig(Strategy):
    def __init__(
        self,
        n_drop: int = 10,
        query_batch_size=256,
        unlabeled_pool_size=256,
        test_set_size: int = 512,
        temperature: float = 1.0,
    ):
        self.eps = 1e-6
        self.n_drop = n_drop
        self.batch_size = query_batch_size
        self.unlabeled_pool_size = unlabeled_pool_size
        self.test_set_size = test_set_size
        self.temperature = temperature
        self.is_calibrate = False
        self.n_samples_test = None

    def build_dataloaders(self, dataset):
        # Candidate set
        _, candidate_set = dataset.get_unlabeled_data()
        candidate_idx_subset = torch.randperm(len(candidate_set))[
            : self.unlabeled_pool_size
        ]
        candidate_subset = Subset(candidate_set, candidate_idx_subset)

        # Test set
        test_set = dataset.get_test_data()
        test_idx_subset = torch.randperm(len(test_set))[: self.test_set_size]
        test_subset = Subset(test_set, test_idx_subset)

        # Dataloaders
        candidate_loader = DataLoader(
            candidate_subset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_subset, batch_size=self.batch_size, num_workers=0, shuffle=True
        )
        return candidate_loader, test_loader

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> Tensor:
        combined_inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]

        logprobs_list = []  # [N + N_t, K, Cl]
        for _ in range(self.n_drop):
            # [N + N_t, Cl]
            logprobs = self.conditional_predict(
                combined_inputs, self.n_samples_test, independent=False
            )
            logprobs_list.append(logprobs.cpu().unsqueeze(1))

        logprobs = torch.cat(logprobs_list, dim=1)  # [N + N_t, K, Cl]
        epig_fn = epig_from_logprobs_using_matmul if use_matmul else epig_from_logprobs

        return epig_fn(logprobs[: len(inputs)], logprobs[len(inputs) :])  # [N,]

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ):
        # features = self.net(inputs, n_model_samples)  # [N, K, Cl] TODO
        features, _ = self.net(inputs)  # [N, K, Cl] TODO
        return log_softmax(features, dim=-1)  # [N, K, Cl]

    def estimate_epig(
        self, loader: DataLoader, target_inputs: Tensor, use_matmul: bool
    ) -> Dictionary:
        scores = Dictionary()

        candidate_idxs = []
        for inputs, _, candidate_idxs_b in loader:
            epig_scores = self.estimate_epig_minibatch(
                inputs, target_inputs, use_matmul
            )  # [B,]
            scores.append({"epig": epig_scores.cpu()})
            candidate_idxs.append(candidate_idxs_b)

        candidate_idxs = torch.hstack(candidate_idxs)
        return scores.concatenate(), candidate_idxs

    def query(self, n, net, dataset):
        torch.set_grad_enabled(False)

        self.net = net
        self.net.train()  # To enable dropout

        candidate_loader, test_loader = self.build_dataloaders(dataset)
        target_inputs, _, _ = next(iter(test_loader))

        scores, idx_candidates = self.estimate_epig(
            candidate_loader, target_inputs, use_matmul=False
        )

        scores = scores["epig"]
        min_x_score, min_x_idx = scores.sort(
            descending=True, axis=-1
        )  # Choose with the maximial score
        choesn_idxs = idx_candidates[min_x_idx[:n]]

        self.net.eval()
        return choesn_idxs
