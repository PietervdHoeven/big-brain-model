import torch
import torch.nn as nn

from big_brain.models.blocks import ConvLayer, DeconvLayer

# class Decoder(nn.Module):
#     def __init__(self, norm: str = "group", activation: str = "gelu"):
#         super().__init__()
#         # [B, 512, 1, 1, 1]
#         self.unbottle1 = DeconvLayer(
#             512, 512, kernel_size=(2,3,2), stride=1, padding=0, activation=activation, norm=norm)
#         self.unbottle2 = DeconvLayer(
#             512, 256, kernel_size=3, stride=1, padding=0, activation=activation, norm=norm)
#         self.unbottle3 = DeconvLayer(
#             256, 128, kernel_size=3, stride=1, padding=0, activation=activation, norm=norm)
        
#         # [B, 256, 6, 7, 6]
#         self.up1 = DeconvLayer(
#             128, 128, kernel_size=2, stride=2, padding=0, activation=activation, norm=norm)
#         self.conv1 = ConvLayer(
#             128, 64, pool=None, activation=activation, norm=norm)
        
#         # [B, 128, 12, 14, 12]
#         self.up2 = DeconvLayer(64, 64, kernel_size=2, stride=2, padding=0, activation=activation, norm=norm)
#         self.conv2 = ConvLayer(64, 32, pool=None, activation=activation, norm=norm)

#         # [B, 64, 24, 28, 24]
#         self.up3 = DeconvLayer(32, 32, kernel_size=2, stride=2, padding=0, activation=activation, norm=norm)
#         self.conv3 = ConvLayer(32, 16, pool=None, activation=activation, norm=norm)

#         # [B, 32, 48, 56, 48]
#         self.up4 = DeconvLayer(16, 16, kernel_size=2, stride=2, padding=0, activation=activation, norm=norm)
#         self.conv4 = ConvLayer(16, 1, pool=None, activation=None, norm=None)

#         # [B, 1, 96, 112, 96]

#     def forward(self,
#                 x: torch.Tensor,
#                 ) -> torch.Tensor:
#         for layer in self._modules.values():
#             x = layer(x)
#             #print(f"After {layer.__class__.__name__}: {x.shape}")
        # return x

class Decoder(nn.Module):
    """
    base_feats : must be the same starting value you used in the Encoder
                 (e.g. 16 or 32).  The network will up-sample in reverse
                 order of the multipliers.
    mults      : same tuple as in Encoder, e.g. (1, 2, 4, 8)
                 The decoder will therefore go
                 128→64→32→16 or 256→128→64→32, etc.
    """
    def __init__(self,
                 base_feats: int = 16,
                 mults: tuple = (1, 2, 4, 8),
                 norm: str = "group",
                 activation: str = "gelu"):
        super().__init__()

        chans = [base_feats * m for m in mults]          # [16, 32, 64, 128]
        chans_rev = chans[::-1]                          # [128, 64, 32, 16]

        self.blocks = nn.ModuleList()

        in_c = chans_rev[0] * 4        # matches Encoder bottleneck (×4)
        # ---- bottleneck “un-bottle” -----------------------
        self.blocks.append(DeconvLayer(in_c,  in_c,  kernel_size=(2,3,2), stride=1,
                                       padding=0, activation=activation, norm=norm))
        self.blocks.append(DeconvLayer(in_c,  in_c//2, kernel_size=3, stride=1,
                                       padding=0, activation=activation, norm=norm))
        self.blocks.append(DeconvLayer(in_c//2, chans_rev[0], kernel_size=3, stride=1,
                                       padding=0, activation=activation, norm=norm))
        in_c = chans_rev[0]

        # ---- up-sampling path -----------------------------
        for out_c in chans_rev[1:]:
            # upsample
            self.blocks.append(DeconvLayer(in_c, in_c, kernel_size=2, stride=2,
                                           padding=0, activation=activation, norm=norm))
            # conv
            self.blocks.append(ConvLayer(in_c, out_c, pool=None,
                                         activation=activation, norm=norm))
            in_c = out_c

        # ---- final 1-channel conv -------------------------
        self.blocks.append(DeconvLayer(in_c, in_c, kernel_size=2, stride=2,
                                       padding=0, activation=activation, norm=norm))
        self.blocks.append(ConvLayer(in_c, 1, pool=None,
                                     activation=None, norm=None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.blocks:
            x = layer(x)
        return x
    
#Example usage
# x = torch.randn(16, 512, 1, 1, 1)  # Example input tensor
# decoder = Decoder()
# output = decoder(x)
# print(f"Output shape: {output.shape}")  # Should be [B, 1, 96, 112, 96]