# big_brain/models/factories.py
import hydra
import torch.nn as nn

class ConvPipeBuilder(nn.Module):
    """
    Instantiate a sequential path from a list of Hydra block specs.
    """
    def __init__(self, block_list):
        super().__init__()
        print(f"Building ConvPipe with {len(block_list)} blocks")
        print(f"Block list: {block_list}")
        self.net = nn.Sequential(*block_list)

    def forward(self, x):
        return self.net(x)