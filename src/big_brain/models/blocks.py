import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    """
    Regular 3-D convolution -> norm -> activation -> (optional) dropout.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = "batch",            # "batch" | "inst" | "group"
        activation: str = "relu",       # "relu"  | "gelu"
        dropout: float = 0.0,
        pool: str | None = "max",              # "max" | "avg" | None
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        norm_layer = {
            "batch": nn.BatchNorm3d,
            "inst":  nn.InstanceNorm3d,
            "group": lambda c: nn.GroupNorm(8, c),
            None: nn.Identity,
        }[norm]
        self.norm = norm_layer(out_ch)

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            None: nn.Identity,
        }[activation]
        self.act = act_layer()

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        pool_layer = {
            "max": nn.MaxPool3d,
            "avg": nn.AvgPool3d,
            None: nn.Identity,
        }[pool]
        self.pool = pool_layer(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return self.pool(x)
    

class DepthConvLayer(nn.Module):
    """
    Depthwise-separable 3-D convolution:
      - depthwise conv (groups=in_ch)
      - pointwise 1x1x1 conv
    Follows with norm, activation, dropout the same way.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        norm: str = "batch",            # "batch" | "inst" | "group"
        activation: str = "relu",       # "relu"  | "gelu"
        dropout: float = 0.0,
        pool: str | None = None,              # "max" | "avg" | None
    ):
        super().__init__()

        self.depthwise = nn.Conv3d(
            in_ch,
            in_ch,
            kernel_size,
            stride,
            padding,
            groups=in_ch,
        )
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1, bias=False)

        norm_layer = {
            "batch": nn.BatchNorm3d,
            "inst":  nn.InstanceNorm3d,
            "group": lambda c: nn.GroupNorm(8, c),
        }[norm]
        self.norm = norm_layer(out_ch)

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[activation]
        self.act = act_layer()

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        pool_layer = {
            "max": nn.MaxPool3d,
            "avg": nn.AvgPool3d,
            None: nn.Identity,
        }[pool]
        self.pool = pool_layer(kernel_size=2, stride=2, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return self.pool(x)
    

class DeconvLayer(nn.Module):
    """
    Transposed convolution layer with optional norm, activation, and dropout.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        norm: str = "batch",            # "batch" | "inst" | "group"
        activation: str = "relu",       # "relu"  | "gelu"
        dropout: float = 0.0,
    ):
        super().__init__()

        self.deconv = nn.ConvTranspose3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        norm_layer = {
            "batch": nn.BatchNorm3d,
            "inst":  nn.InstanceNorm3d,
            "group": lambda c: nn.GroupNorm(8, c),
        }[norm]
        self.norm = norm_layer(out_ch)

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[activation]
        self.act = act_layer()

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x
    


class TokenEmbedder(nn.Module):
    """
    (latent z 512, gradient g 4) --> d_model
    """
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.proj_z = nn.Linear(512, d_model)
        self.proj_g = nn.Linear(4, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, g: torch.Tensor):
        """
        z: [B, 512] latent vector
        g: [B, 4] gradient info (bval, bvec, bvec, bvec)
        """
        z_emb = self.proj_z(z)
        g_emb = self.proj_g(g)
        x = z_emb + g_emb       # [B, d_model]
        x = self.layernorm(x)   # [B, d_model]
        return x
    

class TiedLinear(nn.Module):
    """
    Acts like nn.Linear(d_model -> 512, bias=False)
    but re-uses the *column* weight matrix of the embedder
    (shape (d_model, 512)) and applies it transposed.
    """
    def __init__(self, shared_weight: nn.Parameter):
        super().__init__()
        self.weight = shared_weight            # shape (d_model, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear expects weight shape (out_features, in_features)
        return F.linear(x, self.weight.t())    # (B,L,512)