import torch.nn as nn
import torch.nn.functional as F

class SlimEncoder01(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.pad_input = lambda x: F.pad(x, (0,1, 0,1, 0,1))

        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2, ceil_mode=True)
        )                                  # → 32×46×55×46

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2, ceil_mode=True)
        )                                  # → 64×23×28×23

        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, stride=2, padding=1),  # stride-2 conv
            nn.BatchNorm3d(128), nn.ReLU(inplace=True)
        )                                  # → 128×12×14×12

        self.block4 = nn.Sequential(
            nn.Conv3d(128, 64, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, stride=2, padding=1),    # another learned down-sample
            nn.BatchNorm3d(64), nn.ReLU(inplace=True)
        )                                  # → 64×6×7×6

        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 32, 1), nn.BatchNorm3d(32), nn.ReLU(inplace=True)
        )                                  # → 32×6×7×6

        self.fc = nn.Linear(32 * 6 * 7 * 6, z_dim)

    def forward(self, x):
        x = self.pad_input(x)   # (B,1,92,110,92)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.bottleneck(x)
        x = x.flatten(1)
        z = self.fc(x)          # (B,256)
        return z