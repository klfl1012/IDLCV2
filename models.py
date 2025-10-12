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

    def __init__(self, num_classes=None, in_channels=1):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d((1, 1, 1))

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
    def __init__(self, num_classes, in_channels=3, num_frames=8):
        super().__init__()
        self.backbone = Simple2DCNN(num_classes=None, in_channels=in_channels * num_frames)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        feats = self.backbone(x)
        return self.fc(feats)


class LateFusion2D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Feature extractors (no classification head)
        self.rgb_model = Simple2DCNN(num_classes=None)

        # Final fusion classifier
        self.fc = nn.Linear(10 * 256, num_classes)

    def forward(self, rgb, flow):
        B, T, C, H, W = rgb.shape

        # Flatten temporal dimension
        rgb = rgb.view(B * T, C, H, W)

        # Extract per-frame features
        rgb_feats = self.rgb_model(rgb).view(B, T, -1).mean(dim=1)

        # Fuse modalities
        return self.fc(rgb_feats)

class TwoStream2D(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.rgb_model = Simple2DCNN(num_classes=10)
        self.flow_model = Simple2DCNN(num_classes=10)

    def forward(self, rgb, flow):
        B, T, C, H, W = rgb.shape

        # Flatten temporal dimension
        rgb = rgb.view(B * T, C, H, W)
        flow = flow.view(B, (T - 1) * C, H, W)

        #softmax on both streams
        rgb_feats = self.rgb_model(rgb).view(B, T, -1)
        flow_feats = self.flow_model(flow).view(B, T - 1, -1)

        rgb_feats = F.softmax(rgb_feats, dim=-1)
        flow_feats = F.softmax(flow_feats, dim=-1)

        return rgb_feats + flow_feats

