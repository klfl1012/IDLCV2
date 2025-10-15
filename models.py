import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple2DCNN(nn.Module):

    def __init__(self, num_classes=None, in_channels=3):
        super(Simple2DCNN, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.fc1:
            return self.fc1(x)
        return x
    

class Simple3DCNN(nn.Module):

    def __init__(self, num_classes=None, in_channels=3):
        super(Simple3DCNN, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.fc1:
            return self.fc1(x)
        return x
  
    

class FrameAggregation2D(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_frames=10):
        super().__init__()
        self.backbone = Simple2DCNN(num_classes=None, in_channels=in_channels * num_frames)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        feats = self.backbone(x)
        return self.fc(feats)


class LateFusion2D(nn.Module):

    def __init__(self, num_classes: int, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = Simple2DCNN(num_classes=None)
        self.classifier = nn.Linear(2560, num_classes)

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

    def __init__(self, num_classes):
        super().__init__()
        self.rgb_model = Simple2DCNN(num_classes=10)
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

