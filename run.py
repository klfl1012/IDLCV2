from litmodel import LitClassifier, get_trainer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


pl.seed_everything(21)

X_train, y_train = torch.randn(100, 10), torch.randint(0, 2, (100,))
X_val, y_val = torch.randn(20, 10), torch.randint(0, 2, (20,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)


model = nn.Module()
criterion = nn.CrossEntropyLoss()
optimizer_class = torch.optim.AdamW
optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-2}
scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5, "verbose": True}
outdir = "./results"


litmodel = LitClassifier(
    model=model,
    criterion=criterion,
    optimizer_class=optimizer_class,  
    optimizer_params=optimizer_kwargs,
    scheduler_class=scheduler_class,
    scheduler_params=scheduler_params,
    outdir=outdir
)

trainer = get_trainer(
    max_epochs=100,
    early_stopping_patience=10,
    outdir=outdir
)

trainer.fit(litmodel, train_loader, val_loader)