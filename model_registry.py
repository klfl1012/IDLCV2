from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Sequence

import torch

from models import (
    FrameAggregation2D,
    LateFusion2D,
    Simple2DCNN,
    Simple3DCNN,
    TwoStream2D,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build_fn: Callable[..., torch.nn.Module]
    input_type: str
    requires_flow: bool = False
    description: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "simple2dcnn": ModelSpec(
        name="Simple2DCNN",
        build_fn=lambda num_classes, _num_frames, **extras: Simple2DCNN(num_classes=num_classes, **extras),
        input_type="frames_flat",
        description="2D CNN that processes frames independently.",
    ),
    "simple3dcnn": ModelSpec(
        name="Simple3DCNN",
        build_fn=lambda num_classes, _num_frames, **extras: Simple3DCNN(num_classes=num_classes, **extras),
        input_type="clip_3d",
        description="3D CNN operating on spatio-temporal volumes.",
    ),
    "frameaggregation2d": ModelSpec(
        name="FrameAggregation2D",
        build_fn=lambda num_classes, num_frames, **extras: FrameAggregation2D(
            num_classes=num_classes,
            num_frames=num_frames,
            **extras,
        ),
        input_type="clip",
        description="Aggregates frame features with a 2D backbone.",
    ),
    "latefusion2d": ModelSpec(
        name="LateFusion2D",
        build_fn=lambda num_classes, num_frames, **extras: LateFusion2D(
            num_classes=num_classes,
            num_frames=num_frames,
            **extras,
        ),
        input_type="late_fusion",
        requires_flow=False,
        description="Late fusion of per-frame RGB features without optical flow.",
    ),
    "twostream2d": ModelSpec(
        name="TwoStream2D",
        build_fn=lambda num_classes, _num_frames, **extras: TwoStream2D(num_classes=num_classes, **extras),
        input_type="two_stream",
        requires_flow=True,
        description="Two-stream architecture combining RGB and flow predictions.",
    ),
}


def available_models() -> Tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def resolve_model(name: str) -> ModelSpec:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available options: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key]


def build_collate_fn(input_type: str) -> Callable[[Sequence[Tuple[Dict[str, torch.Tensor], torch.Tensor]]], Tuple[Any, torch.Tensor]]:
    # -> Callable[[Tuple[Dict[str, torch.Tensor], ...]], Tuple[Any, torch.Tensor]]:
    def _collate_fn(batch) -> Tuple[Any, torch.Tensor]:
        rgb = torch.stack([item["rgb"] for item in batch], dim=0)
        labels = torch.stack([item["label"] for item in batch], dim=0)

        if input_type in {"frames_flat"}:
            batch_size, num_frames, channels, height, width = rgb.shape
            merged = rgb.view(batch_size * num_frames, channels, height, width)
            inputs = {
                "frames": merged,
                "batch_size": batch_size,
                "num_frames": num_frames,
            }
            return inputs, labels

        if input_type == "clip":
            return rgb, labels

        if input_type == "clip_3d":
            rgb_3d = rgb.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
            return rgb_3d, labels

        if input_type == "two_stream":
            missing_flow = [idx for idx, item in enumerate(batch) if "flow" not in item]
            if missing_flow:
                raise ValueError(
                    "Optical flow data is required for this model but missing for samples: "
                    + ", ".join(map(str, missing_flow))
                )

            flow = torch.stack([item["flow"] for item in batch], dim=0)

            if flow.size(1) > 1:
                flow = flow[:, :-1, ...]
            return (rgb, flow), labels

        if input_type == "late_fusion":
            batch_size, num_frames, channels, height, width = rgb.shape
            rgb_flat = rgb.view(batch_size * num_frames, channels, height, width)
            inputs = {
                "rgb": rgb_flat,
                "batch_size": batch_size,
                "num_frames": num_frames,
            }
            return inputs, labels

        raise ValueError(f"Unsupported input_type '{input_type}'.")

    return _collate_fn
