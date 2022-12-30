import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet18
import types

logger = logging.getLogger(__name__)


class MNIST_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.max_pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.max_pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(1024, 128)
        self.drop3 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x_in, temperature=1.0):
        x = self.conv1(x_in)
        x = self.drop1(x)
        x = self.max_pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.max_pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 1024)
        x = self.fc1(x)
        e1 = self.drop3(x)
        y = self.fc2(e1 / temperature)
        return y, e1

    def get_embedding_dim(self):
        return 128

    def get_classifer(self):
        return self.fc2


class EMNIST_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.max_pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.max_pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.drop3 = nn.Dropout(cfg.dropout)
        self.max_pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(128, 512)
        self.drop4 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(512, 47)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.drop1(x)
        x = self.max_pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.max_pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.drop3(x)
        x = self.max_pool3(x)
        x = self.relu3(x)

        x = x.view(-1, 128)
        x = self.fc1(x)
        e1 = self.drop4(x)
        y = self.fc2(e1)
        return y, e1

    def get_embedding_dim(self):
        return 47

    def get_classifer(self):
        return self.fc2


class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.leaky_relu(self.fc1(x))
        e1 = F.leaky_relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

    def get_classifer(self):
        return self.fc3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


def CIFAR10_Net(cfg):
    resnet18_handle = resnet18(pretrained=True, progress=True)

    def forward_with_dropout(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = torch.nn.functional.dropout(x, p=cfg.dropout, training=self.training)
        x = self.layer2(x)
        x = torch.nn.functional.dropout(x, p=cfg.dropout, training=self.training)
        x = self.layer3(x)
        x = torch.nn.functional.dropout(x, p=cfg.dropout, training=self.training)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        e = torch.nn.functional.dropout(x, p=cfg.dropout, training=self.training)
        y = self.fc(e / temperature)

        return y, e

    num_classes = 10
    fc = resnet18_handle.fc
    classifier = nn.Linear(fc.in_features, num_classes)
    resnet18_handle.fc = classifier
    resnet18_handle.forward = types.MethodType(forward_with_dropout, resnet18_handle)
    resnet18_handle.get_embedding_dim = types.MethodType(
        lambda self: 512, resnet18_handle
    )
    resnet18_handle.get_classifer = types.MethodType(
        lambda self: self.fc, resnet18_handle
    )
    return resnet18_handle
