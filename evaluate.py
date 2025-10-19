import argparse
import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from dataloader import UCFLikeDataset, load_ucf_metadata_split
from litmodel import LitClassifier
from model_registry import available_models, build_collate_fn, resolve_model


def _move_to_device(data: Union[torch.Tensor, dict, tuple, list], device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) if isinstance(v, (torch.Tensor, dict, list, tuple)) else v for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(x, device) for x in data)
    if isinstance(data, list):
        return [_move_to_device(x, device) for x in data]
    return data


def evaluate(
    checkpoint_path: Path,
    model_name: str,
    dataset_root: str,
    num_classes: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    use_flow: bool = False,
    flow_format: str = "npy",
    input_type: str = "frames_flat",
    pretrained_vgg: bool = False,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    manifest = load_ucf_metadata_split(dataset_root, "test", require_flow=use_flow)
    dataset = UCFLikeDataset(
        manifest,
        num_frames=num_frames,
        use_flow=use_flow,
        flow_format=flow_format,
        return_tensor_format="TCHW",
    )

    collate_fn = build_collate_fn(input_type)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model_spec = resolve_model(model_name)
    model_kwargs = {}
    if model_spec.name.lower() in {"simple2dcnn", "simple3dcnn", "latefusion2d", "twostream2d", "frameaggregation2d"}:
        model_kwargs["pretrained_vgg"] = pretrained_vgg
    model = model_spec.build_fn(num_classes, num_frames, **model_kwargs)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.AdamW
    optimizer_params = {"lr": 1e-3, "weight_decay": 1e-2}
    scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5}

    lit_model = LitClassifier.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params,
        outdir=str(checkpoint_path.parent),
    )

    lit_model.to(device)
    lit_model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = _move_to_device(inputs, device)
            labels = labels.to(device, non_blocking=True)

            logits = lit_model(inputs)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "samples": total_samples,
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "model": model_name,
            "num_classes": num_classes,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "use_flow": use_flow,
            "flow_format": flow_format,
            "pretrained_vgg": pretrained_vgg,
        }
        with save_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Simple2DCNN on the test split.")
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
        help="Model architecture to evaluate (defined in models.py).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/best_checkpoint.ckpt",
        help="Path to the trained Lightning checkpoint.",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of target classes.")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames per clip.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size (clips).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--use-flow",
        action="store_true",
        help="If set, evaluation expects optical flow inputs in the manifest.",
    )
    parser.add_argument(
        "--flow-format",
        type=str,
        default="npy",
        choices=["npy", "png"],
        help="Storage format for optical flow frames.",
    )
    parser.add_argument(
        "--pretrained-vgg",
        action="store_true",
        help="Use ImageNet-pretrained VGG weights for applicable CNN backbones.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_spec = resolve_model(args.model)
    use_flow = args.use_flow or model_spec.requires_flow
    metrics = evaluate(
        checkpoint_path=Path(args.checkpoint),
        model_name=args.model,
        dataset_root=args.dataset_root,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_flow=use_flow,
        flow_format=args.flow_format,
        input_type=model_spec.input_type,
        pretrained_vgg=args.pretrained_vgg,
        device=device,
        save_path=args.metrics_out,
    )

    print("Evaluation results:")
    print(f"  samples:  {metrics['samples']}")
    print(f"  loss:     {metrics['loss']:.4f}")
    print(f"  accuracy: {metrics['accuracy'] * 100:.2f}%")
    if args.metrics_out is not None:
        print(f"  saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()
