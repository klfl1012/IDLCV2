import os 
from typing import Optional, Type
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import PyTorchProfiler


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


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        X, y = batch[:2]
        logits = self(X)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return {"loss": loss, "acc": acc}
    

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        avg_val_loss = torch.stack([x["acc"] for x in outputs]).mean().item()
        self.history["train_loss"].append(avg_train_loss)
        self.history["train_acc"].append(avg_val_loss)


    def validation_step(self, batch, batch_idx):
        X, y = batch[:2]
        logits = self(X)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return {"val_loss": loss, "val_acc": acc}
    

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean().item()
        self.history["val_loss"].append(avg_val_loss)
        self.history["val_acc"].append(avg_val_acc)
        if "train_loss" in self.history:
            self.history["train_loss"].append(0)
            self.history["train_acc"].append(0)


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

    profiler = PyTorchProfiler(
        dirpath=outdir,
        filename="profiler_trace.json",
        schedule=None,
        activities=["cpu", "cuda"],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[checkpoint, early_stop_cb],
        log_every_n_steps=10,
        logger=csv_logger,
        default_root_dir=outdir,
        profiler=profiler
    )

    return trainer