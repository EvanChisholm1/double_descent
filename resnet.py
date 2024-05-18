import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=0, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
        
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        out = out + shortcut
    
        return self.relu3(out)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.res_layers = nn.Sequential(
            ResBlock(64, 64, 3, 1),
            ResBlock(64, 128, 3, 1),
            ResBlock(128, 128, 3, 1),
            ResBlock(128, 256, 3, 1),
            ResBlock(256, 512, 3, 1),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.bn1(x))

        x = self.res_layers(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
