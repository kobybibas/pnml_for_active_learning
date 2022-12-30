import logging
import time
from typing import Tuple

import numpy as np
import torch
import wandb
from torch.nn.functional import cross_entropy
from torch.utils import data as data
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)


mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
fashion_mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
)
svhn_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ]
)
cifar10_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)


class Data:
    def __init__(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        t0 = time.time()

        self.device = device
        self.n_pool, self.n_val, self.n_test = len(X_train), len(X_val), len(X_test)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        # Plot x5
        for _ in range(5):
            wandb_mnist_plot_images(X_train, title="Training set")
            wandb_mnist_plot_images(X_val, title="Validation set")
            wandb_mnist_plot_images(X_test, title="Test set")

        # Propogate trought dataloader to have vectors transformed vectors to the experimnet
        self.X_train, self.Y_train = X_train.to(self.device), Y_train.to(self.device)
        self.X_val, self.Y_val = X_val.to(self.device), Y_val.to(self.device)
        self.X_test, self.Y_test = X_test.to(self.device), Y_test.to(self.device)

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
        labeled_idxs = np.arange(self.n_pool)[
            self.labeled_idxs & (self.Y_train.cpu().numpy() != -1)
        ]
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
        return 1.0 * (self.Y_test.to(preds.device) == preds).sum().item() / self.n_test

    def cal_test_loss(self, probs: torch.Tensor) -> float:
        return cross_entropy(probs, self.Y_test.to(probs.device)).item()

    def count_ind_training_labels(self) -> torch.Tensor:
        training_labels = self.get_labeled_labels().cpu()
        training_labels = training_labels[training_labels != -1]
        label_counts = torch.tensor(
            [
                torch.sum(training_labels == label)
                for label in torch.unique(training_labels)
            ]
        )
        return label_counts


def wandb_mnist_plot_images(data_tensor, title="MNIST Images"):
    perm = torch.randperm(data_tensor.size(0))
    idx = perm[:64]
    image_array = make_grid(
        data_tensor[idx],
        nrow=8,
        padding=2,
        pad_value=0,
    )
    images = wandb.Image(image_array)
    wandb.log({title: images})


def extract_dataset_to_tensors(
    raw_dataset: torch.utils.data.Dataset, limit: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    data, targets = [], []
    for x, y in raw_dataset:
        data.append(x)
        targets.append(y)
    data = torch.vstack(data)[:limit]
    targets = torch.tensor(targets)[:limit]

    if len(data.shape) == 3:
        # For MNIST like dataset, add channel dimension
        data = data.unsqueeze(1)
    return data, targets


def execute_train_val_split(
    train_data: torch.Tensor,
    train_targets: torch.Tensor,
    validation_set_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    val_data = train_data[-validation_set_size:]
    val_targets = train_targets[-validation_set_size:]
    train_data = train_data[:-validation_set_size]
    train_targets = train_targets[:-validation_set_size]
    return train_data, train_targets, val_data, val_targets


def get_MNIST(
    data_dir: str = "../data", validation_set_size: int = 1024, dataset_limit: int = -1
) -> Data:
    raw_train = datasets.MNIST(
        data_dir, train=True, download=True, transform=mnist_transforms
    )
    raw_test = datasets.MNIST(
        data_dir, train=False, download=True, transform=mnist_transforms
    )

    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, validation_set_size
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)
    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_MNIST_C(
    data_dir: str = "../data",
    validation_set_size: int = 1024,
    dataset_limit: int = -1,
) -> Data:

    raw_train = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.RandomErasing(p=0.5, scale=(0.05, 0.1)),
            ]
        ),
    )

    raw_test = datasets.MNIST(
        data_dir, train=False, download=True, transform=mnist_transforms
    )

    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, validation_set_size
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)

    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_MNIST_OOD(
    data_dir: str = "../data", validation_set_size: int = 1024, dataset_limit: int = -1
) -> Data:

    # Train set
    raw_train = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=mnist_transforms,
    )

    raw_ood = datasets.FashionMNIST(
        data_dir,
        train=False,
        download=True,
        transform=mnist_transforms,
    )

    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, validation_set_size
    )
    ood_data, _ = extract_dataset_to_tensors(raw_ood, dataset_limit)
    train_data = torch.vstack((train_data, ood_data))
    train_targets = torch.hstack(
        (train_targets, -1 * torch.ones(ood_data.shape[0]))
    ).long()

    # Test set
    raw_test = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=mnist_transforms,
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)

    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_EMNIST(
    data_dir: str = "../data", val_set_size: int = 1024, dataset_limit: int = -1
) -> Data:
    raw_train = datasets.EMNIST(data_dir, split="balanced", train=True, download=True)
    raw_test = datasets.EMNIST(data_dir, split="balanced", train=False, download=True)

    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, val_set_size
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)
    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_FashionMNIST(
    data_dir: str = "../data",
    val_set_size: int = 1024,
    dataset_limit: int = -1,
) -> Data:
    raw_train = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=fashion_mnist_transforms
    )
    raw_test = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=fashion_mnist_transforms
    )
    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, val_set_size
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)
    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_SVHN(
    data_dir: str = "../data",
    val_set_size: int = 1024,
    dataset_limit: int = -1,
) -> Data:
    raw_train = datasets.SVHN(
        data_dir, split="train", download=True, transform=svhn_transforms
    )
    raw_test = datasets.SVHN(
        data_dir, split="test", download=True, transform=svhn_transforms
    )
    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, val_set_size
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)
    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_CIFAR10(
    data_dir: str = "../data", val_set_size: int = 1024, dataset_limit: int = -1
) -> Data:

    raw_train = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=cifar10_transforms
    )
    raw_test = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=cifar10_transforms
    )

    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, val_set_size
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)
    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_CIFAR10_OOD(
    data_dir: str = "../data", validation_set_size: int = 1024, dataset_limit: int = -1
) -> Data:

    # Train set
    raw_train = datasets.CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=cifar10_transforms,
    )

    raw_ood = datasets.SVHN(
        data_dir,
        train=False,
        download=True,
        transform=cifar10_transforms,
    )

    train_data, train_targets = extract_dataset_to_tensors(raw_train, dataset_limit)
    train_data, train_targets, val_data, val_targets = execute_train_val_split(
        train_data, train_targets, validation_set_size
    )
    ood_data, _ = extract_dataset_to_tensors(raw_ood, dataset_limit)
    train_data = torch.vstack((train_data, ood_data))
    train_targets = torch.hstack(
        (train_targets, -1 * torch.ones(ood_data.shape[0]))
    ).long()

    # Test set
    raw_test = datasets.CIFAR10(
        data_dir,
        train=False,
        download=True,
        transform=cifar10_transforms,
    )
    test_data, test_targets = extract_dataset_to_tensors(raw_test, dataset_limit)

    return Data(
        train_data,
        train_targets,
        val_data,
        val_targets,
        test_data,
        test_targets,
    )


def get_dataloaders(
    dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_labels = dataset.get_labeled_labels().cpu()
    train_labels = train_labels[train_labels != -1]

    train_dataset = dataset.get_labeled_data()[-1]
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
