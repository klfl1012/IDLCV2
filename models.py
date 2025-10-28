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
    temporal_init: str = "avg",              # "avg" oder "center" (nur 3D relevant)
    copy_bn_running_stats: Optional[bool] = None,  # None => 2D: True, 3D: False
) -> None:
    if not pretrained_vgg:
        return

    if copy_bn_running_stats is None:
        copy_bn_running_stats = not is_3d

    # VGG laden (kompatibel mit neueren/älteren torchvision Versionen)
    try:
        weights_enum = getattr(models, f"{vgg_variant.upper()}_Weights")
        weights = getattr(weights_enum, "DEFAULT", getattr(weights_enum, "IMAGENET1K_V1"))
        vgg = getattr(models, vgg_variant)(weights=weights)
    except Exception:
        vgg = getattr(models, vgg_variant)(pretrained=True)

    vgg_convs = [m for m in vgg.features if isinstance(m, nn.Conv2d)]
    vgg_features = list(vgg.features)
    print("Available VGG convs:", [(m.in_channels, m.out_channels) for m in vgg_convs])

    target_convs_list = list(target_convs)
    target_bns_list = list(target_bns) if target_bns is not None else [None] * len(target_convs_list)

    for idx, (dst, dst_bn) in enumerate(zip(target_convs_list, target_bns_list)):
        # Match nach Out-Channels (und grob nach In-Channels ab Layer 2)
        matched = None
        for candidate_conv in vgg_convs:
            cond = candidate_conv.out_channels == dst.out_channels
            if idx > 0:
                cond &= candidate_conv.in_channels <= dst.in_channels
            if cond:
                matched = candidate_conv
                break
        if matched is None:
            print(f"[WARN] No matching VGG conv found for target[{idx}] (in={dst.in_channels}, out={dst.out_channels})")
            continue

        with torch.no_grad():
            w2d = matched.weight  # (out, in, kh, kw)

            if is_3d:
                # Zeit-Inflation
                k_t = dst.weight.shape[2]
                if temporal_init == "center":
                    w3d = torch.zeros_like(dst.weight)
                    center = k_t // 2
                    in_min = min(w2d.shape[1], dst.in_channels)
                    w3d[:, :in_min, center, :, :] = w2d[:, :in_min, :, :]
                else:  # "avg"
                    w3d = w2d.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / k_t

                # In-Channel-Mismatch (z. B. erste Conv3d mit in_channels != 3)
                if w3d.shape[1] != dst.in_channels:
                    expanded = torch.zeros_like(dst.weight)
                    for c in range(dst.in_channels):
                        expanded[:, c, :, :, :] = w3d[:, c % w3d.shape[1], :, :, :]
                    weight = expanded
                else:
                    weight = w3d
            else:
                # 2D: In-Channel-Mismatch (z. B. erste Conv2d mit in_channels != 3)
                if w2d.shape[1] != dst.in_channels:
                    expanded = torch.zeros_like(dst.weight)
                    for c in range(dst.in_channels):
                        expanded[:, c, :, :] = w2d[:, c % w2d.shape[1], :, :]
                    weight = expanded
                else:
                    weight = w2d

            dst.weight.copy_(weight)
            if matched.bias is not None and dst.bias is not None and matched.bias.shape == dst.bias.shape:
                dst.bias.copy_(matched.bias)
            print(f"Conv[{idx}] copied: weight {tuple(dst.weight.shape)}, bias {tuple(dst.bias.shape) if dst.bias is not None else None}")

        # BN-Parameter übernehmen
        if dst_bn is not None:
            matched_bn = None
            conv_idx = vgg_features.index(matched)
            for j in range(conv_idx + 1, len(vgg_features)):
                if isinstance(vgg_features[j], nn.BatchNorm2d):
                    matched_bn = vgg_features[j]
                    break
                if isinstance(vgg_features[j], nn.Conv2d):
                    break
            if matched_bn is not None:
                with torch.no_grad():
                    if hasattr(dst_bn, "weight") and hasattr(matched_bn, "weight") and dst_bn.weight is not None and matched_bn.weight is not None and dst_bn.weight.shape == matched_bn.weight.shape:
                        assert isinstance(dst_bn.weight, torch.Tensor)
                        assert isinstance(matched_bn.weight, torch.Tensor)
                        dst_bn.weight.copy_(matched_bn.weight)
                    if hasattr(dst_bn, "bias") and hasattr(matched_bn, "bias") and dst_bn.bias is not None and matched_bn.bias is not None and dst_bn.bias.shape == matched_bn.bias.shape:
                        assert isinstance(dst_bn.bias, torch.Tensor)
                        assert isinstance(matched_bn.bias, torch.Tensor)
                        dst_bn.bias.copy_(matched_bn.bias)
                    if copy_bn_running_stats:
                        if hasattr(dst_bn, "running_mean") and hasattr(matched_bn, "running_mean") and dst_bn.running_mean is not None and matched_bn.running_mean is not None and dst_bn.running_mean.shape == matched_bn.running_mean.shape:
                            assert isinstance(dst_bn.running_mean, torch.Tensor)
                            assert isinstance(matched_bn.running_mean, torch.Tensor)
                            dst_bn.running_mean.copy_(matched_bn.running_mean)
                        if hasattr(dst_bn, "running_var") and hasattr(matched_bn, "running_var") and dst_bn.running_var is not None and matched_bn.running_var is not None and dst_bn.running_var.shape == matched_bn.running_var.shape:
                            assert isinstance(dst_bn.running_var, torch.Tensor)
                            assert isinstance(matched_bn.running_var, torch.Tensor)
                            dst_bn.running_var.copy_(matched_bn.running_var)

    # Freeze
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



def _unfreeze_conv_bn_pair(
    conv_bn_pairs: list[tuple[nn.Module, Optional[nn.Module]]],
    freeze_bn_running_stats: bool = True
) -> None:
    """
    Unfreeze multiple conv + BN layer pairs.

    Args:
        conv_bn_pairs: list of tuples (conv_layer, bn_layer). bn_layer can be None.
        freeze_bn_running_stats: if True, keeps BN running stats frozen (calls eval() on BN).
    """
    for conv_layer, bn_layer in conv_bn_pairs:
        # Unfreeze conv
        for p in conv_layer.parameters():
            p.requires_grad = True
        # Unfreeze BN if given
        if bn_layer is not None:
            for p in bn_layer.parameters():
                p.requires_grad = True
            if freeze_bn_running_stats and isinstance(bn_layer, (nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layer.eval()
        print(f"[UNFREEZE] Unfroze conv: {conv_layer.__class__.__name__}, "
              f"BN: {bn_layer.__class__.__name__ if bn_layer is not None else 'None'}"
              f"{' (BN running stats frozen)' if freeze_bn_running_stats else ''}")


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
        if pretrained_vgg:
            _unfreeze_conv_bn_pair(
                conv_bn_pairs=[
                    (self.conv1, self.bn1),
                    (self.conv2, self.bn2),
                    (self.conv3, self.bn3),
                    ],
                freeze_bn_running_stats=False
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
        freeze_backbone: bool = False,
        vgg_variant: str = "vgg16_bn",
        dropout_p: float = 0.2,
        activation: str = "silu",
        norm_type: str = "batch",
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = _select_norm3d(64, norm_type)
        self.pool1 = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout3d(p=dropout_p)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = _select_norm3d(128, norm_type)
        self.pool2 = nn.MaxPool3d(2)
        self.dropout2 = nn.Dropout3d(p=dropout_p)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = _select_norm3d(256, norm_type)
        self.pool3 = nn.AdaptiveAvgPool3d(1)
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
            freeze_backbone=pretrained_vgg,
            is_3d=True,
        )
        if pretrained_vgg:
            _unfreeze_conv_bn_pair(
                conv_bn_pairs=[
                    (self.conv1, self.bn1),
                    (self.conv2, self.bn2),
                    (self.conv3, self.bn3),
                ],
                freeze_bn_running_stats=False
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
