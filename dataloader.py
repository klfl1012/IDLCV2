import csv
import glob
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_DEFAULT_FRAME_SIZE = (112, 112)


def _default_frame_transform() -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([
        transforms.Resize(_DEFAULT_FRAME_SIZE),
        transforms.ToTensor(),
    ])


def _ensure_dir(path: str, description: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Expected {description} at '{path}', but it does not exist.")


class UCFLikeDataset(Dataset):
    """Dataset that loads RGB frames (and optional optical flow) from a manifest."""

    def __init__(
        self,
        manifest: List[Dict],
        num_frames: int = 10,
        use_flow: bool = True,
        flow_format: str = "npy",
        frame_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        flow_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_tensor_format: str = "TCHW",
    ) -> None:
        if not manifest:
            raise ValueError("Manifest is empty; ensure metadata CSV paths are correct.")

        if flow_format not in {"npy", "png"}:
            raise ValueError("flow_format must be either 'npy' or 'png'.")

        if return_tensor_format not in {"TCHW", "CTHW"}:
            raise ValueError("return_tensor_format must be 'TCHW' or 'CTHW'.")

        self.manifest = manifest
        self.num_frames = int(num_frames)
        self.use_flow = bool(use_flow)
        self.flow_format = flow_format
        self.frame_transform = frame_transform or _default_frame_transform()
        self.flow_transform = flow_transform
        self.return_tensor_format = return_tensor_format

    def __len__(self):
        return len(self.manifest)

    def _list_sorted(self, folder: str, pattern: str) -> List[str]:
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in '{folder}'.")
        return files

    def _sample_indices(self, available: int) -> List[int]:
        if available <= 0:
            raise ValueError("No frames available for sample.")

        if available <= self.num_frames:
            idx = list(range(available))
            idx.extend([available - 1] * (self.num_frames - available))
            return idx

        step = available / float(self.num_frames)
        return [int(step * i) for i in range(self.num_frames)]

    def _load_frame_tensor(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        tensor = self.frame_transform(img)
        return tensor

    def _load_flow_tensor(self, path: str) -> torch.Tensor:
        if self.flow_format == "npy":
            arr = np.load(path)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3 and arr.shape[-1] in (2, 3):
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.ndim == 3 and arr.shape[0] in (2, 3):
                pass
            else:
                raise ValueError(f"Unexpected flow array shape {arr.shape} in '{path}'.")
            tensor = torch.from_numpy(arr).float()
        else:
            img = Image.open(path).convert("RGB")
            tensor = transforms.ToTensor()(img)

        if self.flow_transform:
            tensor = self.flow_transform(tensor)
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_info = self.manifest[index]
        frames_dir = sample_info["frames_dir"]

        try:
            frame_files = self._list_sorted(frames_dir, "frame_*.*")
        except FileNotFoundError:
            # fall back to generic image patterns
            try:
                frame_files = self._list_sorted(frames_dir, "*.jpg")
            except FileNotFoundError:
                frame_files = self._list_sorted(frames_dir, "*.png")

        indices = self._sample_indices(len(frame_files))
        rgb_frames = [self._load_frame_tensor(frame_files[i]) for i in indices]
        rgb = torch.stack(rgb_frames, dim=0)

        output: Dict[str, torch.Tensor] = {
            "rgb": rgb,
            "label": torch.tensor(int(sample_info.get("label", -1)), dtype=torch.long),
        }

        if self.use_flow:
            flow_dir = sample_info.get("flow_dir")
            if self.flow_format == "png" and not flow_dir:
                flow_dir = sample_info.get("flow_png_dir")

            if not flow_dir or not os.path.isdir(flow_dir):
                raise FileNotFoundError(
                    f"Optical flow requested but directory not found for sample '{sample_info['video_id']}'."
                )

            pattern = "flow_*.npy" if self.flow_format == "npy" else "flow_*.*"
            flow_files = self._list_sorted(flow_dir, pattern)
            flow_frames = [
                self._load_flow_tensor(flow_files[min(i, len(flow_files) - 1)])
                for i in indices
            ]
            flow = torch.stack(flow_frames, dim=0)
            output["flow"] = flow

        if self.return_tensor_format == "CTHW":
            output["rgb"] = output["rgb"].permute(1, 0, 2, 3)
            if self.use_flow and "flow" in output:
                output["flow"] = output["flow"].permute(1, 0, 2, 3)

        return output


def load_ucf_metadata_split(
    dataset_root: str,
    split: str,
    *,
    require_flow: bool = False,
) -> List[Dict]:
    """Read metadata CSV for a given split and build a manifest list."""
    csv_path = os.path.join(dataset_root, "metadata", f"{split}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at '{csv_path}'.")

    frames_split_root = os.path.join(dataset_root, "frames", split)
    _ensure_dir(frames_split_root, f"frames/{split} directory")

    flows_split_root = os.path.join(dataset_root, "flows", split)
    flows_png_split_root = os.path.join(dataset_root, "flows_png", split)

    manifest: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row.get("action") or row.get("class")
            video_name = row.get("video_name")
            if not action or not video_name:
                raise ValueError("Metadata CSV must contain 'action' and 'video_name' columns.")

            frames_dir = os.path.join(frames_split_root, action, video_name)
            if not os.path.isdir(frames_dir):
                raise FileNotFoundError(f"Frames directory not found: '{frames_dir}'.")

            flow_dir = os.path.join(flows_split_root, action, video_name)
            flow_png_dir = os.path.join(flows_png_split_root, action, video_name)

            flow_dir_exists = os.path.isdir(flow_dir)
            flow_png_dir_exists = os.path.isdir(flow_png_dir)

            if require_flow and not (flow_dir_exists or flow_png_dir_exists):
                raise FileNotFoundError(
                    "Flow data required but missing. Checked: "
                    f"'{flow_dir}' and '{flow_png_dir}'."
                )

            label_value = row.get("label")
            if label_value is None:
                raise ValueError("Metadata CSV must contain a 'label' column with integer class ids.")

            manifest.append(
                {
                    "video_id": video_name,
                    "action": action,
                    "frames_dir": frames_dir,
                    "flow_dir": flow_dir if flow_dir_exists else None,
                    "flow_png_dir": flow_png_dir if flow_png_dir_exists else None,
                    "label": int(label_value),
                }
            )

    if require_flow:
        missing = [
            m["video_id"]
            for m in manifest
            if not (m.get("flow_dir") or m.get("flow_png_dir"))
        ]
        if missing:
            raise FileNotFoundError(
                "Flow data required but missing for videos: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )

    return manifest
