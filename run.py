import argparse
from pathlib import Path
import ast
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import UCFLikeDataset, load_ucf_metadata_split
from litmodel import LitClassifier, get_trainer
from model_registry import available_models, build_collate_fn, resolve_model


def build_dataloader(
	*,
	manifest,
	num_frames: int,
	batch_size: int,
	num_workers: int,
	use_flow: bool,
	flow_format: str,
	input_type: str,
	shuffle: bool,
	is_train: bool = False,
) -> DataLoader:
	dataset = UCFLikeDataset(
		manifest,
		num_frames=num_frames,
		use_flow=use_flow,
		flow_format=flow_format,
		return_tensor_format="TCHW",
		is_train=is_train,
	)

	collate_fn = build_collate_fn(input_type)

	return DataLoader(
		dataset,
		batch_size=batch_size,
	shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_fn,
	)


def describe_batch(inputs, labels, prefix: str) -> None:
	print(prefix)
	if isinstance(inputs, dict):
		for key, tensor in inputs.items():
			if hasattr(tensor, "shape"):
				print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
			else:
				print(f"  {key}: value={tensor}")
	elif isinstance(inputs, (tuple, list)):
		for idx, tensor in enumerate(inputs):
			print(f"  input[{idx}]: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
	else:
		print(f"  inputs: shape={tuple(inputs.shape)}, dtype={inputs.dtype}")
	print(f"  labels: shape={tuple(labels.shape)}, dtype={labels.dtype}")


def build_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a video model on UCF-like data.")
	parser = argparse.ArgumentParser(description="Train a video model on UCF-like data.")
	parser.add_argument("--num-classes", type=int, default=10, help="Number of target classes.")
	parser.add_argument("--num-frames", type=int, default=10, help="Frames sampled per clip.")
	parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (clips).")
	parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
	parser.add_argument("--outdir", type=str, default="./results", help="Output directory for logs and checkpoints.")
	parser.add_argument("--seed", type=int, default=21, help="Random seed for reproducibility.")
	parser.add_argument("--max-epochs", type=int, default=5, help="Maximum training epochs.")
	parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use (adamw or sgd).")
	parser.add_argument("--optimizer-kwargs", type=str, default="{'lr': 3e-4, 'weight_decay': 1e-2}", help="Optimizer keyword arguments as a dictionary string.")
	parser.add_argument("--scheduler", type=str, default="reduce_on_plateau", help="Learning rate scheduler to use (reduce_on_plateau or step).")
	parser.add_argument("--scheduler-kwargs", type=str, default="{}", help="Scheduler keyword arguments as a dictionary string.")
	parser.add_argument(
		"--dataset-root",
		type=str,
		default="/dtu/datasets1/02516/ucf101_noleakage",
		help="Root directory containing frames/flows and metadata CSVs.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="simple2dcnn",
		choices=available_models(),
		help="Model architecture to train (defined in models.py).",
	)
	parser.add_argument(
		"--flow-format",
		type=str,
		default="npy",
		choices=["npy", "png"],
		help="Storage format for optical flow frames.",
	)
	parser.add_argument(
		"--use-flow",
		action="store_true",
		help="Force optical flow loading even for models that do not require it.",
	)
	parser.add_argument(
		"--early-stopping-patience",
		type=int,
		default=10,
		help="Early stopping patience on validation loss.",
	)
	parser.add_argument(
		"--no-preview",
		action="store_true",
		help="Disable printing of example batch shapes before training.",
	)
	parser.add_argument(
		"--pretrained-vgg",
		action="store_true",
		help="Use ImageNet-pretrained VGG weights for applicable CNN backbones.",
	)
	parser.add_argument(
		"--pretrained-r2p1d",
		action="store_true",
		help="Use Kinetics-pretrained R2+1D weights for applicable backbones.",
	)
	parser.add_argument(
		"--pretrained-resnet",
		action="store_true",
		help="Use ImageNet-pretrained ResNet weights for applicable CNN backbones.",
	)

	return parser.parse_args()


def parse_dict_arg(arg_value: str, arg_name: str) -> dict:
	try:
		result = ast.literal_eval(arg_value)
		if not isinstance(result, dict):
			raise ValueError()
		return result
	except Exception:
		raise ValueError(f"{arg_name} must be a dictionary string, e.g., \"{{'lr': 0.001, 'weight_decay': 0.01}}\"")


def get_optimizer_class(name: str):
	name_lower = name.lower()
	mapping = {
		"adamw": "AdamW",
		"sgd": "SGD",
	}
	if name_lower not in mapping:
		raise ValueError("Unsupported optimizer. Choose 'adamw' or 'sgd'.")
	return getattr(torch.optim, mapping[name_lower])


def get_scheduler_class(name: str):
	name = name.lower()
	schedulers = {
		"reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
		"step": torch.optim.lr_scheduler.StepLR,
		"cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
	}
	if name not in schedulers:
		raise ValueError("Unsupported scheduler. Choose 'reduce_on_plateau' or 'step'.")
	return schedulers[name]





def main():
	args = build_args()

	pl.seed_everything(args.seed, workers=True)

	spec = resolve_model(args.model)
	use_flow = args.use_flow or spec.requires_flow

	train_manifest = load_ucf_metadata_split(args.dataset_root, "train", require_flow=use_flow)
	val_manifest = load_ucf_metadata_split(args.dataset_root, "val", require_flow=use_flow)

	train_loader = build_dataloader(
		manifest=train_manifest,
		num_frames=args.num_frames,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		use_flow=use_flow,
		flow_format=args.flow_format,
		input_type=spec.input_type,
		shuffle=True,
		is_train=True,
	)

	val_loader = build_dataloader(
		manifest=val_manifest,
		num_frames=args.num_frames,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		use_flow=use_flow,
		flow_format=args.flow_format,
		input_type=spec.input_type,
		shuffle=False,
	)

	if not args.no_preview:
		train_inputs, train_labels = next(iter(train_loader))
		describe_batch(train_inputs, train_labels, "Preview train batch:")

		val_inputs, val_labels = next(iter(val_loader))
		describe_batch(val_inputs, val_labels, "Preview val batch:")

	model_kwargs = {}
	if spec.name.lower() in {"simple2dcnn", "simple3dcnn", "latefusion2d", "twostream2d", "frameaggregation2d"}:
		model_kwargs["pretrained_vgg"] = args.pretrained_vgg
	
	if spec.name.lower() in {"frameaggregation2d_resnet", "latefusion2d_resnet", "twostream2d_mod"}:
		model_kwargs["pretrained_resnet"] = args.pretrained_resnet
	
	if spec.name.lower() == "r2p1d":
		model_kwargs["pretrained_r2p1d"] = args.pretrained_r2p1d

		
	model = spec.build_fn(args.num_classes, args.num_frames, **model_kwargs)
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

	optimizer_kwargs = parse_dict_arg(args.optimizer_kwargs, "optimizer-kwargs")
	scheduler_kwargs = parse_dict_arg(args.scheduler_kwargs, "scheduler-kwargs")
	optimizer_class = get_optimizer_class(args.optimizer)
	scheduler_class = get_scheduler_class(args.scheduler)

	if not scheduler_kwargs:
		defaults = {
			"step": {"step_size": 10, "gamma": 0.1},
			"reduce_on_plateau": {"mode": "min", "factor": 0.1, "patience": 5},
			"cosine": {"T_max": 20},
		}
		scheduler_kwargs = defaults.get(args.scheduler.lower(), {})

	litmodel = LitClassifier(
		model=model,
		criterion=criterion,
		optimizer_class=optimizer_class,
		optimizer_params=optimizer_kwargs,
		scheduler_class=scheduler_class,
		scheduler_params=scheduler_kwargs,
		outdir=args.outdir,
	)

	trainer = get_trainer(
		max_epochs=args.max_epochs,
		early_stopping_patience=args.early_stopping_patience,
		outdir=args.outdir,
	)

	trainer.fit(litmodel, train_loader, val_loader)


if __name__ == "__main__":
	main()