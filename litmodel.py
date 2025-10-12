import os
from typing import Optional, Type

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity


class LitClassifier(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_params: dict,
        scheduler_class: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None,
        scheduler_params: Optional[dict] = None,
        outdir: str = "./results"
    ):
        super().__init__()
        self.model = model 
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self._train_epoch_metrics = []
        self._val_epoch_metrics = []
    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self._train_epoch_metrics = []

    def on_validation_epoch_start(self):
        self._val_epoch_metrics = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )

        self._train_epoch_metrics.append((loss.detach(), acc.detach()))
        return loss

    def on_train_epoch_end(self):
        if not self._train_epoch_metrics:
            return
        losses = torch.stack([item[0] for item in self._train_epoch_metrics]).mean().item()
        accs = torch.stack([item[1] for item in self._train_epoch_metrics]).mean().item()
        self.history["train_loss"].append(losses)
        self.history["train_acc"].append(accs)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )

        self._val_epoch_metrics.append((loss.detach(), acc.detach()))
        return loss

    def on_validation_epoch_end(self):
        if not self._val_epoch_metrics:
            return
        losses = torch.stack([item[0] for item in self._val_epoch_metrics]).mean().item()
        accs = torch.stack([item[1] for item in self._val_epoch_metrics]).mean().item()
        self.history["val_loss"].append(losses)
        self.history["val_acc"].append(accs)


    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer



def get_trainer(
        max_epochs: int = 100,
        gpus: int = 1,
        early_stopping_patience: int = 5,
        outdir: str = "./results"
) -> pl.Trainer:
    
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=outdir,
        filename=f"best_checkpoint",
        save_top_k=1,
        mode="min"
    ) 

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min"
    )

    csv_logger = CSVLogger(save_dir=outdir, name="logs")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profiler = PyTorchProfiler(
        dirpath=outdir,
        filename="profiler_trace.json",
        schedule=None,
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[checkpoint, early_stop_cb],
        log_every_n_steps=10,
        logger=csv_logger,
        default_root_dir=outdir,
        profiler=profiler,
    )

    return trainer