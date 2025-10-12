from litmodel import LitClassifier, get_trainer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models import Simple2DCNN

# New dataset utilities
from dataloader import UCFLikeDataset, load_ucf_metadata_split


pl.seed_everything(21)

# Build manifests for train/val splits from metadata CSVs
dataset_root = "/dtu/datasets1/02516/ucf101_noleakage"
USE_FLOW = False  # Set to False to train without optical flow inputs
FLOW_FORMAT = "npy"  # "npy" or "png" depending on your flow storage

train_manifest = load_ucf_metadata_split(dataset_root, "train", require_flow=USE_FLOW)
val_manifest = load_ucf_metadata_split(dataset_root, "val", require_flow=USE_FLOW)

# Create datasets
NUM_FRAMES = 10
BATCH_SIZE = 8

train_dataset = UCFLikeDataset(
	train_manifest,
	num_frames=NUM_FRAMES,
	use_flow=USE_FLOW,
	flow_format=FLOW_FORMAT,
	return_tensor_format="TCHW",
)

val_dataset = UCFLikeDataset(
	val_manifest,
	num_frames=NUM_FRAMES,
	use_flow=USE_FLOW,
	flow_format=FLOW_FORMAT,
	return_tensor_format="TCHW",
)


def frames_to_images_collate(batch):
	rgb = torch.stack([item["rgb"] for item in batch], dim=0)
	labels = torch.stack([item["label"] for item in batch], dim=0)

	batch_size, num_frames, channels, height, width = rgb.shape
	rgb = rgb.view(batch_size * num_frames, channels, height, width)
	labels = labels.unsqueeze(1).expand(-1, num_frames).reshape(-1)

	return rgb, labels


train_loader = DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=4,
	collate_fn=frames_to_images_collate,
)
val_loader = DataLoader(
	val_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=4,
	collate_fn=frames_to_images_collate,
)
train_inputs, train_labels = next(iter(train_loader))
print("Flattened train batch:")
print(f"  inputs: shape={tuple(train_inputs.shape)}, dtype={train_inputs.dtype}")
print(f"  labels: shape={tuple(train_labels.shape)}, dtype={train_labels.dtype}")

val_inputs, val_labels = next(iter(val_loader))
print("Flattened val batch:")
print(f"  inputs: shape={tuple(val_inputs.shape)}, dtype={val_inputs.dtype}")
print(f"  labels: shape={tuple(val_labels.shape)}, dtype={val_labels.dtype}")


NUM_CLASSES = 10
model = Simple2DCNN(num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer_class = torch.optim.AdamW
optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-2}
scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5}
outdir = "./results"


litmodel = LitClassifier(
	model=model,
	criterion=criterion,
	optimizer_class=optimizer_class,
	optimizer_params=optimizer_kwargs,
	scheduler_class=scheduler_class,
	scheduler_params=scheduler_params,
	outdir=outdir,
)

trainer = get_trainer(
	max_epochs=5,
	early_stopping_patience=10,
	outdir=outdir,
)

trainer.fit(litmodel, train_loader, val_loader)