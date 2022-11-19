import logging
import time
from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    TensorDataset,
    WeightedRandomSampler,
)
from torchvision import datasets
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Data:
    def __init__(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        handler: Dataset,
        device: str = None,
    ):
        t0 = time.time()
        self.handler = handler

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_pool, self.n_val, self.n_test = len(X_train), len(X_val), len(X_test)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        # Propogate trought dataloader to have vectors transformed vectors to the experimnet
        self.X_train, self.Y_train = self.transfor_set(X_train, Y_train, self.device)
        self.X_val, self.Y_val = self.transfor_set(X_val, Y_val, self.device)
        self.X_test, self.Y_test = self.transfor_set(X_test, Y_test, self.device)

        self.train_dataset = TensorDataset(
            self.X_train,
            self.Y_train,
            torch.arange(len(self.X_train)),
        )
        self.val_dataset = TensorDataset(
            self.X_val,
            self.Y_val,
            torch.arange(len(self.X_val)),
        )
        self.test_dataset = TensorDataset(
            self.X_test,
            self.Y_test,
            torch.arange(len(self.X_test)),
        )

        logger.info(f"Train. [Data Labels]=[{self.X_train.shape} {self.Y_train.shape}]")
        logger.info(f"Val. [Data Labels]=[{self.X_val.shape} {self.Y_val.shape}]")
        logger.info(f"Test. [Data Labels]=[{self.X_test.shape} {self.Y_test.shape}]")
        logger.info(f"Data __init__ in {time.time()-t0:.1f} sec")

    def transfor_set(
        self, x_data: torch.Tensor, y_labels: torch.Tensor, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_h = self.handler(x_data, torch.tensor(y_labels).unsqueeze(1))
        loader = DataLoader(dataset_h, shuffle=False, batch_size=128, num_workers=0)

        X_transformed, Y_transformed = [], []
        for x, y, _ in tqdm(loader):
            X_transformed.append(x)
            Y_transformed.append(y)

        X_transformed = torch.vstack(X_transformed).to(device)
        Y_transformed = torch.vstack(Y_transformed).squeeze().to(device)

        return X_transformed, Y_transformed

    def initialize_labels(self, num: int):
        # Generate initial labeled pool. Make sure equal number of labels in each class
        num_labels = len(self.Y_train.unique())
        init_train_per_label = max(int(num / num_labels), 1)
        for label in range(num_labels):
            label_idxs = np.where(self.Y_train.cpu() == label)[0]
            np.random.shuffle(label_idxs)
            self.labeled_idxs[label_idxs[:init_train_per_label]] = True

    def get_labeled_labels(self) -> torch.Tensor:
        return self.Y_train[self.labeled_idxs]

    def get_labeled_data(self) -> Tuple[np.array, Subset]:
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, Subset(self.train_dataset, labeled_idxs)

    def get_unlabeled_data(self) -> Tuple[np.array, TensorDataset]:
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, Subset(self.train_dataset, unlabeled_idxs)

    def get_train_data(self) -> Tuple[np.array, TensorDataset]:
        return self.labeled_idxs.copy(), self.train_dataset

    def get_val_data(self) -> Tuple[np.array, TensorDataset]:
        return self.val_dataset

    def get_test_data(self) -> TensorDataset:
        return self.test_dataset

    def cal_test_acc(self, preds) -> float:
        return 1.0 * (self.Y_test.to(preds.device) == preds).sum().item() / self.n_test

    def cal_test_loss(self, probs: torch.Tensor) -> float:
        return cross_entropy(probs, self.Y_test.to(probs.device)).item()


def get_MNIST(
    handler, data_dir: str = "../data", validation_set_size: int = 1024
) -> Data:
    raw_train = datasets.MNIST(data_dir, train=True, download=True)
    raw_test = datasets.MNIST(data_dir, train=False, download=True)

    train_data = raw_train.data[:-validation_set_size]
    train_targets = raw_train.targets[:-validation_set_size]
    val_data = raw_train.data[-validation_set_size:]
    val_targets = raw_train.targets[-validation_set_size:]
    test_data = raw_test.data
    test_targets = raw_test.targets

    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
        handler,
    )


def get_EMNIST(
    handler, data_dir: str = "../data", validation_set_size: int = 1024
) -> Data:
    raw_train = datasets.EMNIST(data_dir, split="balanced", train=True, download=True)
    raw_test = datasets.EMNIST(data_dir, split="balanced", train=False, download=True)

    train_data = raw_train.data[:-validation_set_size]
    train_targets = raw_train.targets[:-validation_set_size]
    val_data = raw_train.data[-validation_set_size:]
    val_targets = raw_train.targets[-validation_set_size:]
    test_data = raw_test.data
    test_targets = raw_test.targets

    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
        handler,
    )


def get_FashionMNIST(handler, data_dir: str = "../data") -> Data:
    raw_train = datasets.FashionMNIST(data_dir, train=True, download=True)
    raw_test = datasets.FashionMNIST(data_dir, train=False, download=True)
    return Data(
        raw_train.data[:40000],
        raw_train.targets[:40000],
        raw_test.data[:40000],
        raw_test.targets[:40000],
        handler,
    )


def get_SVHN(handler, data_dir: str = "../data") -> Data:
    data_train = datasets.SVHN(data_dir, split="train", download=True)
    data_test = datasets.SVHN(data_dir, split="test", download=True)
    return Data(
        data_train.data[:40000],
        torch.from_numpy(data_train.labels)[:40000],
        data_test.data[:40000],
        torch.from_numpy(data_test.labels)[:40000],
        handler,
    )


def get_CIFAR10(
    handler, data_dir: str = "../data", validation_set_size: int = 1024
) -> Data:

    raw_train = datasets.CIFAR10(data_dir, train=True, download=True)
    raw_test = datasets.CIFAR10(data_dir, train=False, download=True)

    train_data = raw_train.data[:-validation_set_size]
    train_targets = raw_train.targets[:-validation_set_size]
    val_data = raw_train.data[-validation_set_size:]
    val_targets = raw_train.targets[-validation_set_size:]
    test_data = raw_test.data
    test_targets = raw_test.targets

    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
        handler,
    )


def get_dataloaders(
    dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = dataset.get_labeled_data()[-1]

    if False:
        # Upsample sparse data
        label_bin_count = dataset.get_labeled_labels().bincount()
        class_weights = label_bin_count.sum() / label_bin_count
        sample_weights = [0] * len(train_dataset)
        for idx, (_, label, _) in enumerate(train_dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight.cpu().item()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            # shuffle=True,
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            sampler=sampler,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=RandomFixedLengthSampler(train_dataset, 5056),
        batch_size=batch_size,
        num_workers=0,
    )  # Align with https://github.com/BlackHC/BatchBALD/blob/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0/src/run_experiment.py#L205

    val_loader = DataLoader(
        dataset.get_val_data(),
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset.get_test_data(),
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


import torch
from torch.utils import data as data


class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset: data.Dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(range(len(self.dataset)))

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length
