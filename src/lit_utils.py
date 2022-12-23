import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

logger = logging.getLogger(__name__)


def initalize_trainer(
    cfg: DictConfig,
    wandb_logger,
    enable_progress_bar: bool = False,
    is_lr_monitor: bool = False,
    out_dir: str = None,
):
    callbacks = [
        EarlyStopping(
            monitor="loss/val",
            mode="min",
            patience=cfg.early_stopping_patience,
        )
    ]
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        save_top_k=1,
        monitor="loss/val",
        mode="min",
        filename="best.pth",
    )
    callbacks.append(checkpoint_callback)
    if is_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        default_root_dir=out_dir,
        enable_checkpointing=True,
        gpus=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=1 if enable_progress_bar else 1000,
        callbacks=callbacks,
    )
    return trainer, checkpoint_callback


class LitClassifier(pl.LightningModule):
    def __init__(self, net, cfg, device: str, class_weight: torch.Tensor = None):
        self.cfg = cfg
        self.device_ = device
        self.save_hyperparameters()
        super().__init__()

        self.cfg = cfg
        self.clf = net(cfg)

        # Loss
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, x, temperature=1.0):
        return self.clf(x, temperature=temperature)

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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.cfg.reduce_on_plature_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/val",
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

    def T_scaling(self, logits, temperature):
        return torch.div(logits, temperature)

    def calibrate(self, val_loader: DataLoader):
        self.clf.eval()
        device = self.device

        temperature = nn.Parameter(torch.ones(1).cuda())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [temperature], lr=0.001, max_iter=10000, line_search_fn="strong_wolfe"
        )

        logits_list, labels_list = [], []
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                _, logits = self.clf(images)
                logits_list.append(logits)
                labels_list.append(labels)

        # Create tensors
        logits_list = torch.vstack(logits_list)
        labels_list = torch.hstack(labels_list)

        def _eval():
            loss = criterion(self.T_scaling(logits_list, temperature), labels_list)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return temperature.cpu().detach().item()
