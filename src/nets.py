import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Tuple


class Net:
    def __init__(self, net, params, device: str):
        self.net = net
        self.params = params
        self.device = device
        self.net_init = self.net()

    def train(self, data):
        n_epoch = self.params["n_epoch"]
        self.clf = copy.deepcopy(self.net_init).to(self.device).half()
        self.clf.train()
        optimizer = optim.SGD(
            self.clf.parameters(), lr=self.params.lr, momentum=self.params.momentum
        )

        loader = DataLoader(
            data, shuffle=True, batch_size=self.params.batch_size, num_workers=0,
        )
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out.float(), y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.params.batch_size_test, num_workers=0
        )
        preds = []
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds.append(pred)
        return torch.hstack(preds).int().cpu()

    def predict_prob(self, data):
        self.clf.eval()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.params.batch_size_test, num_workers=0,
        )
        with torch.no_grad():
            probs = []
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs.append(prob)
        return torch.vstack(probs).float().cpu()

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(
            data, shuffle=False, batch_size=self.params.batch_size_test, num_workers=0
        )
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data: Dataset, n_drop: int = 10):
        self.clf.train()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.params.batch_size_test, num_workers=0
        )

        probs_drop = []
        for i in range(n_drop):
            probs = []
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs.append(prob.cpu())
                    # probs[i][idxs] += F.softmax(out, dim=1).cpu()
            probs_drop.append(torch.vstack(probs).unsqueeze(0))
        probs_drop = torch.vstack(probs_drop).float()
        return probs_drop

    def get_embeddings(self, data) -> torch.Tensor:
        self.clf.eval()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.params.batch_size_test, num_workers=0
        )

        embeddings = []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(self.device)
                _, e1 = self.clf(x)
                embeddings.append(e1)
        return torch.vstack(embeddings)

    def get_emb_logit_prob(
        self, data, is_norm_features: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.clf.eval()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.params.batch_size_test, num_workers=0
        )
        clf_last_layer = self.clf.get_classifer()
        embeddings, logits, probs = [], [], []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(self.device)
                out, e1 = self.clf(x)

                # Normalize
                if is_norm_features:
                    norm = torch.linalg.norm(e1, dim=-1, keepdim=True)
                    e1 = e1 / norm
                    # Forward with feature normalization
                    out = clf_last_layer(e1)

                embeddings.append(e1)
                logits.append(out)
                prob = F.softmax(out, dim=1)
                probs.append(prob)
        return torch.vstack(embeddings), torch.vstack(logits), torch.vstack(probs)


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

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
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
