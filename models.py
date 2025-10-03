import torch
import torch.nn as nn




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
        self.pool3 = nn.MaxPool2d((1, 1))

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
    



