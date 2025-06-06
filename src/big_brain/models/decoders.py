import torch.nn as nn

class SlimDecoder01(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 32 * 6 * 7 * 6)  # Match encoder output shape

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )  # → 12 × 14 × 12

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )  # → 23 × 27 × 23

        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # → 46 × 55 × 46

        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # → 92 × 110 × 92

        self.final = nn.Conv3d(32, 1, kernel_size=3, padding=1)  # Keep size

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 6, 7, 6)  # Reshape to volume
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)  # Now 92 × 110 × 92

        # Crop back to original shape
        x = x[:, :, 0:91, 0:109, 0:91]
        return x
