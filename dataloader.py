import os
import glob
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class UCFLikeDataset(Dataset):
    """Dataset for UCF-like folder layout with frames, flows and metadata.

    Expected directory layout (root):
      frames/            # per-video folders or flat frames named <video>_frame_0001.jpg
      flows/             # numpy arrays per video or per-frame flow files
      flows_png/         # optional flow visualizations as png pairs
      metadata/          # optional csv/json mapping video->label
      videos/            # optional raw videos

    This dataset intentionally keeps I/O simple and leans on the caller to
    provide a `manifest` (list of dicts) describing samples. Each manifest
    entry should contain at least:
      {"video_id": "v_ApplyEyeMakeup_g01_c01", "label": 3, "frames_dir": "/.../frames/v_..."}

    Args:
        manifest: list of dicts describing samples (video_id, frames_dir, label, optional flow_dir)
        num_frames: number of frames to sample per clip
        modality: "rgb", "flow", or "both"
        flow_mode: "npy" (numpy flow arrays), "png" (flow visualizations) or "pseudo" (compute from images)
        transforms: torchvision transforms applied per-frame (to PIL Image)
        return_tensor_format: "TCHW" (T,C,H,W) or "CTHW" (C,T,H,W)
    """

    def __init__(
        self,
        manifest: List[dict],
        num_frames: int = 10,
        modality: str = "both",
        flow_mode: str = "npy",
        transforms: Optional[Callable] = None,
        return_tensor_format: str = "TCHW",
    ):
        self.manifest = manifest
        self.num_frames = int(num_frames)
        self.modality = modality
        self.flow_mode = flow_mode
        self.transforms = transforms or transforms = transforms = transforms or transforms = transforms
        # default transforms: resize + to tensor
        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ])
        assert return_tensor_format in ("TCHW", "CTHW")
        self.return_tensor_format = return_tensor_format

    def __len__(self):
        return len(self.manifest)

    def _load_frame(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transforms(img)

    def _load_flow_npy(self, path: str) -> torch.Tensor:
        arr = np.load(path)
        # expect HWC or CHW
        if arr.ndim == 3 and arr.shape[2] in (2, 3):
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] in (2, 3):
            pass
        else:
            # fallback: expand dims
            arr = np.expand_dims(arr, 0)
        return torch.from_numpy(arr).float()

    def _list_frames(self, frames_dir: str) -> List[str]:
        # support both per-video folder or flat pattern
        if os.path.isdir(frames_dir):
            ext = "*.jpg"
            files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if not files:
                files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            return files
        # else treat frames_dir as glob pattern
        return sorted(glob.glob(frames_dir))

    def _sample_indices(self, n_available: int) -> List[int]:
        if n_available <= self.num_frames:
            # pad by repeating last
            idx = list(range(n_available)) + [n_available - 1] * (self.num_frames - n_available)
            return idx
        # random uniform sampling
        step = n_available / float(self.num_frames)
        return [int(step * i) for i in range(self.num_frames)]

    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]
        frames_dir = item["frames_dir"]
        frames = self._list_frames(frames_dir)
        indices = self._sample_indices(len(frames))

        rgb_list = [self._load_frame(frames[i]) for i in indices]
        # rgb tensor shape: (T, C, H, W)
        rgb = torch.stack(rgb_list, dim=0)

        out = {"rgb": rgb, "label": torch.tensor(item.get("label", -1), dtype=torch.long)}

        if self.modality in ("flow", "both"):
            if self.flow_mode == "npy":
                flow_dir = item.get("flow_dir")
                if flow_dir is None:
                    # try to infer
                    flow_dir = frames_dir.replace("frames", "flows")
                # assume per-frame npy files named similarly to frames
                flow_files = self._list_frames(flow_dir)
                flow_list = []
                for i in indices:
                    flow_path = flow_files[i] if i < len(flow_files) else flow_files[-1]
                    flow_list.append(self._load_flow_npy(flow_path))
                flow = torch.stack(flow_list, dim=0)
            elif self.flow_mode == "png":
                # flows stored as png-based visualizations
                flow_dir = item.get("flow_dir")
                if flow_dir is None:
                    flow_dir = frames_dir.replace("frames", "flows_png")
                flow_files = self._list_frames(flow_dir)
                flow_list = [self._load_frame(flow_files[i]) for i in indices]
                flow = torch.stack(flow_list, dim=0)
            elif self.flow_mode == "pseudo":
                diffs = rgb[1:] - rgb[:-1]
                pad = torch.zeros_like(rgb[:1])
                flow = torch.cat([pad, diffs.abs()], dim=0)
            else:
                raise ValueError("unknown flow_mode")

            out["flow"] = flow

        # return format
        if self.return_tensor_format == "CTHW":
            # convert (T, C, H, W) -> (C, T, H, W)
            out["rgb"] = out["rgb"].permute(1, 0, 2, 3)
            if "flow" in out:
                out["flow"] = out["flow"].permute(1, 0, 2, 3)

        return out


# Example manifest builder
def build_manifest_from_frames_root(frames_root: str, metadata: Optional[dict] = None) -> List[dict]:
    """Scans frames_root for subfolders and builds a manifest list.

    frames_root can contain per-video subfolders (each with frames). metadata
    is an optional dict mapping video_id -> label.
    """
    manifest = []
    for entry in sorted(os.listdir(frames_root)):
        path = os.path.join(frames_root, entry)
        if os.path.isdir(path):
            video_id = entry
            manifest.append({
                "video_id": video_id,
                "frames_dir": path,
                "label": metadata.get(video_id, -1) if metadata else -1,
            })
    return manifest


def build_manifest_from_ucf_split(dataset_root: str, split: str = "train") -> List[dict]:
    """Build manifest from UCF101-like structure with train/test/val splits.
    
    Expected structure:
    dataset_root/
      frames/train/class1/video1/
      frames/train/class1/video2/
      frames/val/class1/video3/
      flows/train/class1/video1/
      ...
    
    Args:
        dataset_root: path to root (e.g., "/dtu/datasets1/02516/ucf101_noleakage")
        split: "train", "val", or "test"
    
    Returns:
        manifest list with video_id, frames_dir, flow_dir, label (class index)
    """
    frames_split_root = os.path.join(dataset_root, "frames", split)
    flows_split_root = os.path.join(dataset_root, "flows", split)
    
    manifest = []
    class_names = sorted(os.listdir(frames_split_root))
    
    for class_idx, class_name in enumerate(class_names):
        class_frames_dir = os.path.join(frames_split_root, class_name)
        class_flows_dir = os.path.join(flows_split_root, class_name)
        
        if os.path.isdir(class_frames_dir):
            for video_name in sorted(os.listdir(class_frames_dir)):
                video_frames_path = os.path.join(class_frames_dir, video_name)
                video_flows_path = os.path.join(class_flows_dir, video_name)
                
                if os.path.isdir(video_frames_path):
                    manifest.append({
                        "video_id": video_name,
                        "class_name": class_name,
                        "frames_dir": video_frames_path,
                        "flow_dir": video_flows_path if os.path.exists(video_flows_path) else None,
                        "label": class_idx,
                    })
    
    return manifest
