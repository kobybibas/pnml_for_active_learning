import logging
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy

logger = logging.getLogger(__name__)


class LitClassifier(pl.LightningModule):
    def __init__(self, net, cfg, device: str):
        self.cfg = cfg
        self.device_ = device
        self.save_hyperparameters()
        super().__init__()

        self.cfg = cfg
        self.clf = net(cfg)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.clf(x)

    def training_step(self, batch, batch_idx):
        return self._loss_helper(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._loss_helper(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._loss_helper(batch, "test")

    def _loss_helper(self, batch, phase: str = "train"):
        x, y, idxs = batch
        y_hat, _ = self.forward(x)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y)

        self.log(f"loss/{phase}", loss, on_epoch=True, on_step=False)
        self.log(f"acc/{phase}", acc, on_epoch=True, on_step=False)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "max",
            patience=self.cfg.lr_scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "acc/val",
                "strict": True,
                "name": None,
            },
        }

    def predict(self, data):
        self.clf.eval()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.cfg.batch_size_test, num_workers=0
        )
        preds, probs = [], []
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device_), y.to(self.device_)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds.append(pred)

                prob = F.softmax(out, dim=1)
                probs.append(prob)
        return torch.hstack(preds).int().cpu(), torch.vstack(probs).float().cpu()

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(
            data, shuffle=False, batch_size=self.cfg.batch_size_test, num_workers=0
        )
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device_), y.to(self.device_)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data: Dataset, n_drop: int = 10):
        self.clf.train()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.cfg.batch_size_test, num_workers=0
        )

        probs_drop = []
        for i in range(n_drop):
            probs = []
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device_), y.to(self.device_)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs.append(prob.cpu())
            probs_drop.append(torch.vstack(probs).unsqueeze(0))
        probs_drop = torch.vstack(probs_drop).float()
        return probs_drop

    def get_embeddings(self, data) -> torch.Tensor:
        self.clf.eval()
        loader = DataLoader(
            data, shuffle=False, batch_size=self.cfg.batch_size_test, num_workers=0
        )

        embeddings = []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(self.device_)
                _, e1 = self.clf(x)
                embeddings.append(e1)
        return torch.vstack(embeddings)
