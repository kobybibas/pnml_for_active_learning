import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import logging
import time

logger = logging.getLogger(__name__)


class MNIST_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        t0 = time.time()
        self.x_transformed = torch.vstack(
            [self.transform(Image.fromarray(x.numpy(), mode="L")) for x in X]
        ).unsqueeze(1)
        self.x_transformed = (
            self.x_transformed.cuda()
            if torch.cuda.is_available()
            else self.x_transformed
        )
        self.x_transformed = self.x_transformed.half()
        logger.info(f"Moved dataset to cuda in {time.time()-t0:.2f} sec")
        logger.info(
            f"[shape device dtype]=[{self.x_transformed.shape} {self.x_transformed.device} {self.x_transformed.dtype}]"
        )

    def __getitem__(self, index):
        x, y = self.x_transformed[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class SVHN_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                ),
            ]
        )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
