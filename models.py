import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Simple2DCNN(nn.Module):

    def __init__(self, num_classes=None, in_channels=3, pretrained_vgg=False, freeze_backbone=False, vgg_variant="vgg16"):
        super(Simple2DCNN, self).__init__()
        self.pretrained_vgg = pretrained_vgg
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        self.features = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.pool1,
            self.conv2, self.bn2, nn.ReLU(), self.pool2,
            self.conv3, self.bn3, nn.ReLU(), self.pool3
        )

        self.fc1 = nn.Linear(256, num_classes) if num_classes else None

        if self.pretrained_vgg:
            if in_channels % 3 != 0:
                raise ValueError("Pretrained VGG weights require the input channels to be a multiple of 3.")
            try:
                print(f"[Simple2DCNN] Loading pretrained {vgg_variant.upper()} weights from torchvision...")
                weights = getattr(models, f"{vgg_variant.upper()}_Weights").IMAGENET1K_V1
                vgg = getattr(models, vgg_variant)(weights=weights)
            
            except Exception:
                print(f"[Simple2DCNN] Falling back to torchvision pretrained={True} for {vgg_variant}.")
                vgg = getattr(models, vgg_variant)(pretrained=True)

            vgg_convs = [m for m in vgg.features if isinstance(m, nn.Conv2d)]
            target_convs = [self.conv1, self.conv2, self.conv3]

            src_idx = 0
            for idx, dst in enumerate(target_convs):
                matched = None
                while src_idx < len(vgg_convs):
                    candidate = vgg_convs[src_idx]
                    src_idx += 1
                    if idx == 0:
                        # First conv layer: accept any conv with matching output channels.
                        if candidate.out_channels == dst.out_channels:
                            matched = candidate
                            break
                    else:
                        if (
                            candidate.out_channels == dst.out_channels
                            and candidate.in_channels == dst.in_channels
                        ):
                            matched = candidate
                            break

                if matched is None:
                    raise RuntimeError("Unable to find matching VGG conv layer for Simple2DCNN.")

                src_weight = matched.weight.data
                if idx == 0 and dst.weight.shape[1] != src_weight.shape[1]:
                    print(
                        f"[Simple2DCNN] Expanding first conv weights from {src_weight.shape[1]} to {dst.weight.shape[1]} channels by repetition."
                    )
                    repeats = dst.weight.shape[1]
                    expanded = torch.zeros_like(dst.weight.data)
                    for channel in range(repeats):
                        expanded[:, channel, :, :] = src_weight[:, channel % src_weight.shape[1], :, :]
                    dst.weight.data.copy_(expanded)
                else:
                    dst.weight.data.copy_(src_weight)

                if matched.bias is not None and dst.bias is not None:
                    dst.bias.data.copy_(matched.bias.data)

            if freeze_backbone:
                for p in self.features.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.fc1:
            return self.fc1(x)
        return x
    

class Simple3DCNN(nn.Module):

    def __init__(
        self,
        num_classes=None,
        in_channels=3,
        pretrained_vgg: bool = False,
        freeze_backbone: bool = False,
        vgg_variant: str = "vgg16",
    ):
        super(Simple3DCNN, self).__init__()
        self.pretrained_vgg = pretrained_vgg
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.AdaptiveAvgPool3d(1)

        self.features = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.pool1,
            self.conv2, self.bn2, nn.ReLU(), self.pool2,
            self.conv3, self.bn3, nn.ReLU(), self.pool3
        )

        self.fc1 = nn.Linear(256, num_classes) if num_classes else None

        if self.pretrained_vgg:
            if in_channels != 3:
                raise ValueError("Simple3DCNN pretrained VGG initialization requires in_channels=3.")
            try:
                print(f"[Simple3DCNN] Loading pretrained {vgg_variant.upper()} weights from torchvision...")
                weights = getattr(models, f"{vgg_variant.upper()}_Weights").IMAGENET1K_V1
                vgg = getattr(models, vgg_variant)(weights=weights)
            except Exception:
                print(f"[Simple3DCNN] Falling back to torchvision pretrained={True} for {vgg_variant}.")
                vgg = getattr(models, vgg_variant)(pretrained=True)

            vgg_convs = [m for m in vgg.features if isinstance(m, nn.Conv2d)]
            target_convs = [self.conv1, self.conv2, self.conv3]
            print("[Simple3DCNN] Inflating 2D VGG kernels into 3D convolutions.")
            src_idx = 0
            for dst in target_convs:
                matched = None
                while src_idx < len(vgg_convs):
                    candidate = vgg_convs[src_idx]
                    src_idx += 1
                    if (
                        candidate.out_channels == dst.out_channels
                        and candidate.in_channels == dst.in_channels
                    ):
                        matched = candidate
                        break

                if matched is None:
                    raise RuntimeError(
                        "Unable to align VGG convolution weights with Simple3DCNN layers."
                    )

                weight_2d = matched.weight.data  # (out_c, in_c, k, k)
                k_t = dst.weight.shape[2]
                weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / k_t
                dst.weight.data.copy_(weight_3d)
                if matched.bias is not None and dst.bias is not None:
                    dst.bias.data.copy_(matched.bias.data)

            if freeze_backbone:
                for p in self.features.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.fc1:
            return self.fc1(x)
        return x
  
    

class FrameAggregation2D(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_frames=10, pretrained_vgg: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = Simple2DCNN(
            num_classes=None,
            in_channels=in_channels * num_frames,
            pretrained_vgg=pretrained_vgg,
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        feats = self.backbone(x)
        return self.fc(feats)


class LateFusion2D(nn.Module):

    def __init__(self, num_classes: int, num_frames: int, pretrained_vgg: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = Simple2DCNN(num_classes=None, pretrained_vgg=pretrained_vgg)
        self.classifier = nn.Linear(256 * num_frames, num_classes)

    def forward(self, *, rgb: torch.Tensor, batch_size: int, num_frames: int):
        batch_size = int(batch_size)
        num_frames = int(num_frames) if num_frames else self.num_frames

        expected = batch_size * num_frames
        if rgb.size(0) != expected:
            raise ValueError(
                f"LateFusion2D expected {expected} RGB frames (batch_size={batch_size}, "
                f"num_frames={num_frames}), but received tensor with shape {tuple(rgb.shape)}"
            )

        feats = self.backbone(rgb)
        feats = feats.view(batch_size, -1)
        return self.classifier(feats)

class TwoStream2D(nn.Module):

    def __init__(self, num_classes: int = 10, pretrained_vgg: bool = False):
        super().__init__()
        self.rgb_model = Simple2DCNN(num_classes=10, pretrained_vgg=pretrained_vgg)
        self.flow_model = Simple2DCNN(num_classes=10, in_channels=18)

    def forward(self, rgb, flow):
        B, Tr, Cr, Hr, Wr = rgb.shape
        B, Tf, Cf, Hf, Wf = flow.shape
        # print(rgb.shape, flow.shape)

        # Flatten temporal dimension
        rgb = rgb.view(B * Tr, Cr, Hr, Wr)
        flow = flow.view(B, Tf * Cf, Hf, Wf)

        #softmax on both streams
        rgb_feats = self.rgb_model(rgb).view(B, Tr, -1)
        flow_feats = self.flow_model(flow).view(B, -1)

        rgb_feats = F.softmax(rgb_feats, dim=-1)
        #average rgb over time
        rgb_feats = rgb_feats.mean(dim=1)
        flow_feats = F.softmax(flow_feats, dim=-1)
        return rgb_feats + flow_feats

