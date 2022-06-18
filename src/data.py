import logging
import time

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision import datasets
from tqdm import tqdm
from typing import Tuple

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
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        # Propogate trought dataloader to have vectors transformed vectors to the experimnet
        (self.X_train_trans, self.Y_train_trans,) = self.get_transformed_set(
            self.X_train, self.Y_train, self.device
        )
        self.X_val_trans, self.Y_val_trans = self.get_transformed_set(
            self.X_val, self.Y_val, self.device
        )
        self.X_test_trans, self.Y_test_trans = self.get_transformed_set(
            self.X_test, self.Y_test, self.device
        )
        trainset_size, valset_size, testset_size = (
            len(self.X_train_trans),
            len(self.X_val_trans),
            len(self.X_test_trans),
        )

        self.train_dataset = TensorDataset(
            self.X_train_trans, self.Y_train_trans, torch.arange(trainset_size),
        )
        self.val_dataset = TensorDataset(
            self.X_val_trans, self.Y_val_trans, torch.arange(valset_size),
        )
        self.test_dataset = TensorDataset(
            self.X_test_trans, self.Y_test_trans, torch.arange(testset_size),
        )

        logger.info(
            f"Train. [Data Labels]=[{self.X_train_trans.shape} {self.Y_train_trans.shape}]"
        )
        logger.info(
            f"Validation. [Data Labels]=[{self.X_val_trans.shape} {self.Y_val_trans.shape}]"
        )
        logger.info(
            f"Test. [Data Labels]=[{self.X_test_trans.shape} {self.Y_test_trans.shape}]"
        )
        logger.info(f"Data __init__ in {time.time()-t0:.1f} sec")

    def get_transformed_set(
        self, x_data: torch.Tensor, y_labels: torch.Tensor, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_h = self.handler(x_data, y_labels.unsqueeze(1))
        loader = DataLoader(dataset_h, shuffle=False, batch_size=128, num_workers=0)

        X_transformed, Y_transformed = [], []
        for x, y, _ in tqdm(loader):
            X_transformed.append(x)
            Y_transformed.append(y)

        X_transformed = torch.vstack(X_transformed).to(device).half()
        Y_transformed = torch.vstack(Y_transformed).squeeze().to(device)

        return X_transformed, Y_transformed

    def initialize_labels(self, num: int):
        # Generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_labels(self) -> torch.Tensor:
        return self.Y_train_trans[self.labeled_idxs]

    def get_labeled_data(self) -> Tuple[np.array, Subset]:
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, Subset(self.train_dataset, labeled_idxs)

    def get_unlabeled_data(self) -> Tuple[np.array, TensorDataset]:
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, Subset(self.train_dataset, unlabeled_idxs)

    def get_train_data(self) -> Tuple[np.array, TensorDataset]:
        return self.labeled_idxs.copy(), self.train_dataset

    def get_val_data(self) -> TensorDataset:
        return self.val_dataset

    def get_test_data(self) -> TensorDataset:
        return self.test_dataset

    def cal_test_acc(self, preds) -> float:
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

    def cal_test_loss(self, probs: torch.Tensor) -> float:
        return cross_entropy(probs, self.Y_test).item()


def get_MNIST(
    handler, training_set_size: int = 40000, data_dir: str = "../data"
) -> Data:
    raw_train = datasets.MNIST(data_dir, train=True, download=True)
    raw_test = datasets.MNIST(data_dir, train=False, download=True)
    return Data(
        raw_train.data[:training_set_size],
        raw_train.targets[:training_set_size],
        raw_train.data[training_set_size:],
        raw_train.targets[training_set_size:],
        raw_test.data,
        raw_test.targets,
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


def get_CIFAR10(handler, data_dir: str = "../data") -> Data:
    data_train = datasets.CIFAR10(data_dir, train=True, download=True)
    data_test = datasets.CIFAR10(data_dir, train=False, download=True)
    return Data(
        data_train.data[:40000],
        torch.LongTensor(data_train.targets)[:40000],
        data_test.data[:40000],
        torch.LongTensor(data_test.targets)[:40000],
        handler,
    )


def get_dataloaders(dataset, batch_size:int, batch_size_test:int) -> Tuple[DataLoader,DataLoader]:
    train_loader = DataLoader(
            dataset.get_labeled_data()[-1],
            shuffle=True,
            batch_size=batch_size,
            num_workers=0,
        )
    val_loader = DataLoader(
            dataset.get_val_data(),
            shuffle=False,
            batch_size=batch_size_test,
            num_workers=0,
        )
    return train_loader, val_loader