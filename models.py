import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Union, Optional
from torchvision import models

def _select_norm(channels: int, norm_type: str) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "group":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    if norm_type == "layer":
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    return nn.BatchNorm2d(channels)

def _select_norm3d(channels: int, norm_type: str) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type in {"group", "layer"}:
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    return nn.BatchNorm3d(channels)


def _load_vgg_backbone(
    *,
    target_convs: Iterable[Union[nn.Conv2d, nn.Conv3d]],
    target_bns: Optional[Iterable[Optional[nn.Module]]] = None, 
    in_channels: int,
    pretrained_vgg: bool,
    vgg_variant: str,
    freeze_backbone: bool,
    is_3d: bool,
) -> None:
    if not pretrained_vgg:
        return
    if is_3d and in_channels != 3:
        raise ValueError("3D VGGG initialization requires in_channels=3.")
    if not is_3d and in_channels % 3 != 0:
        raise ValueError("VGG weights require in_channels to be a multiple of 3.")
    try:
        weights = getattr(models, f"{vgg_variant.upper()}_Weights").IMAGENET1K_V1
        vgg = getattr(models, vgg_variant)(weights=weights)
    except Exception:
        vgg = getattr(models, vgg_variant)(pretrained=True)

    # Simpler conv matching logic: get all convs in VGG features
    vgg_convs = [m for m in vgg.features if isinstance(m, nn.Conv2d)]
    print("Available VGG convs:", [(m.in_channels, m.out_channels) for m in vgg_convs])
    
    for idx, dst in enumerate(target_convs):
        matched = None
        for candidate_conv in vgg_convs:  # reset search each time
            cond = candidate_conv.out_channels == dst.out_channels
            # Only check in_channels if we're not at the first layer
            if idx > 0:
                cond &= candidate_conv.in_channels <= dst.in_channels
            if cond:
                matched = candidate_conv
                break

        if matched is None:
            print(f"[WARN] No matching VGG conv found for target[{idx}] "
                f"(in={dst.in_channels}, out={dst.out_channels})")
            continue

        with torch.no_grad():
            weight = matched.weight
            if is_3d:
                k_t = dst.weight.shape[2]
                weight = weight.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / k_t
            elif idx == 0 and dst.weight.shape[1] != weight.shape[1]:
                repeats = dst.weight.shape[1]
                expanded = torch.zeros_like(dst.weight)
                for c in range(repeats):
                    expanded[:, c, :, :] = weight[:, c % weight.shape[1], :, :]
                weight = expanded
            dst.weight.copy_(weight)
            if matched.bias is not None and dst.bias is not None:
                dst.bias.copy_(matched.bias)
            # Print conv copy info
            print(
                f"Conv[{idx}] copied: weight shape = {dst.weight.shape}, "
                f"bias shape = {dst.bias.shape if dst.bias is not None else None}"
            )

        print(f"[VGG INIT] Copied weights for conv[{idx}] "
            f"(VGG in={matched.in_channels}, out={matched.out_channels})")

        # The BN mapping/copying logic remains as before (for 2D only)
        if not is_3d and target_bns is not None:
            try:
                dst_bn = list(target_bns)[idx]
            except Exception:
                dst_bn = None
            # Find the corresponding BN in VGG (if present)
            # We'll look for the next BatchNorm2d after this conv in vgg.features
            matched_bn = None
            vgg_features = list(vgg.features)
            conv_idx = vgg_features.index(matched)
            for j in range(conv_idx + 1, len(vgg_features)):
                if isinstance(vgg_features[j], nn.BatchNorm2d):
                    matched_bn = vgg_features[j]
                    break
                if isinstance(vgg_features[j], nn.Conv2d):
                    break
            if matched_bn is not None and dst_bn is not None:
                with torch.no_grad():
                    # Copy affine parameters (weight, bias)
                    if hasattr(dst_bn, "weight") and dst_bn.weight is not None and \
                    hasattr(matched_bn, "weight") and matched_bn.weight is not None:
                        assert isinstance(dst_bn.weight, torch.Tensor)
                        assert isinstance(matched_bn.weight, torch.Tensor)
                        dst_bn.weight.copy_(matched_bn.weight)
                    if hasattr(dst_bn, "bias") and dst_bn.bias is not None and \
                    hasattr(matched_bn, "bias") and matched_bn.bias is not None:
                        assert isinstance(dst_bn.bias, torch.Tensor)
                        assert isinstance(matched_bn.bias, torch.Tensor)
                        dst_bn.bias.copy_(matched_bn.bias)
                    # Copy running statistics (buffers)
                    if hasattr(dst_bn, "running_mean") and dst_bn.running_mean is not None and \
                    hasattr(matched_bn, "running_mean") and matched_bn.running_mean is not None:
                        assert isinstance(dst_bn.running_mean, torch.Tensor)
                        assert isinstance(matched_bn.running_mean, torch.Tensor)
                        dst_bn.running_mean.copy_(matched_bn.running_mean)
                    if hasattr(dst_bn, "running_var") and dst_bn.running_var is not None and \
                    hasattr(matched_bn, "running_var") and matched_bn.running_var is not None:
                        assert isinstance(dst_bn.running_var, torch.Tensor)
                        assert isinstance(matched_bn.running_var, torch.Tensor)
                        dst_bn.running_var.copy_(matched_bn.running_var)
                    # Print BN copy info
                    print(
                        f"BN[{idx}] copied: weight shape = "
                        f"{dst_bn.weight.shape if hasattr(dst_bn, 'weight') and dst_bn.weight is not None else None}, "
                        f"bias shape = "
                        f"{dst_bn.bias.shape if hasattr(dst_bn, 'bias') and dst_bn.bias is not None else None}"
                    )


    # Freeze backbone layers (also freeze BatchNorm if present, this is the case for vgg16_bn)
    if freeze_backbone:
        for i, conv in enumerate(target_convs):
            if not hasattr(conv, "parameters") or not callable(conv.parameters):
                raise TypeError(f"target_convs[{i}] is not a nn.Module: {type(conv)}")
            for p in conv.parameters():
                p.requires_grad = False
            if target_bns is not None:
                try:
                    bn = list(target_bns)[i]
                except Exception:
                    bn = None
                if bn is not None:
                    if not hasattr(bn, "parameters") or not callable(bn.parameters):
                        raise TypeError(f"target_bns[{i}] is not a nn.Module: {type(bn)}")
                    for p in bn.parameters():
                        p.requires_grad = False
                    if isinstance(bn, (nn.BatchNorm2d, nn.BatchNorm3d)):
                        bn.eval()



class Simple2DCNN(nn.Module):
    def __init__(
        self,
        num_classes: int | None = None,
        in_channels: int = 3,
        pretrained_vgg: bool = False,
        freeze_backbone: bool = False,
        vgg_variant: str = "vgg16_bn",
        dropout_p: float = 0.2,
        activation: str = "silu",
        norm_type: str = "batch",
        base_channels: int = 64,
    ):
        super().__init__()
        self.pretrained_vgg = pretrained_vgg
        act_cls = nn.ReLU if activation.lower() == "relu" else nn.SiLU
        widths = [64, 128, 256] if pretrained_vgg else [base_channels, base_channels * 2, base_channels * 4]
        self.feature_dim = widths[-1]
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1)
        self.bn1 = _select_norm(widths[0], norm_type)
        self.conv2 = nn.Conv2d(widths[0], widths[1], kernel_size=3, padding=1)
        self.bn2 = _select_norm(widths[1], norm_type)
        self.conv3 = nn.Conv2d(widths[1], widths[2], kernel_size=3, padding=1)
        self.bn3 = _select_norm(widths[2], norm_type)
        self.features = nn.Sequential(
            self.conv1, self.bn1, act_cls(),
            nn.MaxPool2d(2), nn.Dropout2d(p=dropout_p),
            self.conv2, self.bn2, act_cls(),
            nn.MaxPool2d(2), nn.Dropout2d(p=dropout_p),
            self.conv3, self.bn3, act_cls(),
            nn.AdaptiveAvgPool2d(1), nn.Dropout2d(p=dropout_p),
        )
        self.classifier_dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.feature_dim, num_classes) if num_classes else None
        _load_vgg_backbone(
            target_convs=[self.conv1, self.conv2, self.conv3],
            target_bns=[self.bn1, self.bn2, self.bn3],
            in_channels=in_channels,
            pretrained_vgg=pretrained_vgg,
            vgg_variant=vgg_variant,
            freeze_backbone=pretrained_vgg,
            is_3d=False,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.classifier_dropout(x)
            x = self.fc(x)
        return x

class Simple3DCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        pretrained_vgg: bool = False,
        pretrained_r2p1d: bool = False,
        freeze_backbone: bool = False,
        vgg_variant: str = "vgg16_bn",
        dropout_p: float = 0.2,
        activation: str = "silu",
        norm_type: str = "batch",
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1 = _select_norm3d(64, norm_type)
        self.pool1 = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout3d(p=dropout_p)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = _select_norm3d(128, norm_type)
        self.pool2 = nn.MaxPool3d(2)
        self.dropout2 = nn.Dropout3d(p=dropout_p)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = _select_norm3d(256, norm_type)
        self.pool3 = nn.MaxPool3d(2)
        self.dropout3 = nn.Dropout3d(p=dropout_p)

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.SiLU

        self.features = nn.Sequential(
            self.conv1, self.bn1, act_cls(), self.pool1, self.dropout1,
            self.conv2, self.bn2, act_cls(), self.pool2, self.dropout2,
            self.conv3, self.bn3, act_cls(), self.pool3, self.dropout3,
        )

        self.feature_dim = 256
        self.classifier_dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.feature_dim, num_classes) if num_classes else None
        _load_vgg_backbone(
            target_convs=[self.conv1, self.conv2, self.conv3],
            target_bns=[self.bn1, self.bn2, self.bn3], 
            in_channels=in_channels,
            pretrained_vgg=pretrained_vgg,
            vgg_variant=vgg_variant,
            freeze_backbone=freeze_backbone,
            is_3d=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.classifier_dropout(x)
            x = self.fc(x)
        return x


class ResNet2DVideoCNN(nn.Module):
    """
    Wrapper for torchvision.models.resnet (2D) for video classification.
    Accepts a single frame or a batch of frames in (B, C, H, W) or (B*T, C, H, W).
    """
    def __init__(
        self,
        num_classes: int,
        pretrained_resnet: bool = True,
        freeze_backbone: bool = False,
        resnet_variant: str = "resnet18",
    ):
        super().__init__()
        resnet_fn = getattr(models, resnet_variant)
        weights = None
        if pretrained_resnet:
            try:
                weights_enum = getattr(models, f"{resnet_variant.upper()}_Weights")
                weights = weights_enum.DEFAULT
            except Exception:
                weights = "IMAGENET1K_V1" if pretrained_resnet else None
        self.model = resnet_fn(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class R2p1D(nn.Module):
    """
    Wrapper for torchvision.models.video.r2plus1d_18 for video classification.
    """
    def __init__(
        self,
        num_classes: int,
        pretrained_r2p1d: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        if pretrained_r2p1d:
            try:
                weights = R2Plus1D_18_Weights.DEFAULT
                self.model = r2plus1d_18(weights=weights)
            except Exception:
                self.model = r2plus1d_18(pretrained=True)
        else:
            self.model = r2plus1d_18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith('fc.'):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class FrameAggregation2D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, num_frames: int = 10, pretrained_vgg: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = Simple2DCNN(
            num_classes=None,
            in_channels=in_channels * num_frames,
            pretrained_vgg=pretrained_vgg,
        )
        self.fc = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)
        feats = self.backbone(x)
        return self.fc(feats)

class LateFusion2D(nn.Module):
    def __init__(self, num_classes: int, num_frames: int, pretrained_vgg: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = Simple2DCNN(num_classes=None, pretrained_vgg=pretrained_vgg)
        self.classifier = nn.Linear(self.backbone.feature_dim * num_frames, num_classes)

    def forward(self, *, rgb: torch.Tensor, batch_size: int, num_frames: int):
        batch_size = int(batch_size)
        num_frames = int(num_frames) if num_frames else self.num_frames
        expected = batch_size * num_frames
        if rgb.size(0) != expected:
            raise ValueError(
                f"LateFusion2D erwartete {expected} Frames (batch_size={batch_size}, num_frames={num_frames}), "
                f"erhielt aber {tuple(rgb.shape)}"
            )
        feats = self.backbone(rgb)
        feats = feats.view(batch_size, -1)
        return self.classifier(feats)

class TwoStream2D(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained_vgg: bool = False):
        super().__init__()
        self.rgb_model = Simple2DCNN(num_classes=num_classes, pretrained_vgg=pretrained_vgg)
        self.flow_model = Simple2DCNN(num_classes=num_classes, in_channels=18, pretrained_vgg=False)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, tr, cr, hr, wr = rgb.shape
        b, tf, cf, hf, wf = flow.shape
        rgb = rgb.view(b * tr, cr, hr, wr)
        flow = flow.view(b, tf * cf, hf, wf)
        rgb_feats = self.rgb_model(rgb).view(b, tr, -1)
        flow_feats = self.flow_model(flow).view(b, -1)
        rgb_feats = F.softmax(rgb_feats, dim=-1).mean(dim=1)
        flow_feats = F.softmax(flow_feats, dim=-1)
        return rgb_feats + flow_feats
    

class TwoStream2DModified(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained_resnet: bool = True, use_flow: bool = False):
        super().__init__()
        self.use_flow = use_flow
        # RGB branch
        self.rgb_model = ResNet2DVideoCNN(num_classes=10, pretrained_resnet=pretrained_resnet)
        # Flow branch
        if use_flow:
            self.flow_model = Simple2DCNN(num_classes=None, in_channels=18, pretrained_vgg=False)
        # Dynamically compute classifier input dimension
        flow_dim = self.flow_model.feature_dim if use_flow else 0
        self.classifier = nn.Linear(self.rgb_model.model.fc.in_features + flow_dim, num_classes)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, tr, cr, hr, wr = rgb.shape
        b, tf, cf, hf, wf = flow.shape

        # flatten frames for 2D CNN
        rgb = rgb.view(b * tr, cr, hr, wr)
        rgb_feats = self.rgb_model(rgb)  # (B*T, feat_dim)
        rgb_feats = rgb_feats.view(b, tr, -1).mean(dim=1)  # Aggregate over frames

        if self.use_flow and flow is not None:
            b, tf, cf, hf, wf = flow.shape
            flow = flow.view(b * tf, cf, hf, wf)
            flow_feats = self.flow_model(flow)
            flow_feats = flow_feats.view(b, tf, -1).mean(dim=1)
            combined_feats = torch.cat([rgb_feats, flow_feats], dim=1)
        else:
            combined_feats = rgb_feats

        return self.classifier(combined_feats)


class FrameAggregationResNet(nn.Module):
    def __init__(self, num_classes: int, num_frames: int = 10, pretrained_resnet: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = ResNet2DVideoCNN(num_classes=num_classes, pretrained_resnet=pretrained_resnet)
        self.fc = nn.Linear(self.backbone.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)
        feats = self.backbone(x)
        return self.fc(feats)
    
class LateFusionResNet(nn.Module):
    def __init__(self, num_classes: int, num_frames: int, pretrained_resnet: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = ResNet2DVideoCNN(num_classes=num_classes, pretrained_resnet=pretrained_resnet)
        self.classifier = nn.Linear(self.backbone.model.fc.in_features * num_frames, num_classes)

    def forward(self, *, rgb: torch.Tensor):
        b, t, c, h, w = rgb.shape
        feats = self.backbone(rgb.view(b * t, c, h, w))
        feats = feats.view(b, t * feats.size(1))
        return self.classifier(feats)

