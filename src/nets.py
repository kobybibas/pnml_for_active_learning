import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MNIST_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x_in):
        z = x_in.view(-1, 28 * 28)
        e1 = F.leaky_relu(self.fc1(z))
        y = self.fc2(self.drop1(e1))
        return y, e1

    def get_embedding_dim(self):
        return 50

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


# class CIFAR10_Net(nn.Module):
#     def __init__(self):
#         super(CIFAR10_Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
#         self.fc1 = nn.Linear(1024, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x))
#         x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.leaky_relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 1024)
#         e1 = F.leaky_relu(self.fc1(x))
#         x = F.dropout(e1, training=self.training)
#         x = self.fc2(x)
#         return x, e1

#     def get_embedding_dim(self):
#         return 50

#     def get_classifer(self):
#         return self.fc2


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


class CIFAR10_Net(nn.Module):
    def __init__(self, cfg, block=BasicBlock, num_blocks=None, num_classes=10):
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        super(CIFAR10_Net, self).__init__()
        self.in_planes = 64
        self.num_features = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        e1 = torch.flatten(out, 1)
        out = self.linear(e1)
        return out, e1

    def get_embedding_dim(self):
        return self.num_features

    def get_classifer(self):
        return self.linear


def ResNet18(cfg):
    return CIFAR10_Net(cfg, BasicBlock, [2, 2, 2, 2])

