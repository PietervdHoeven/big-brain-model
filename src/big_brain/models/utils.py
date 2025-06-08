def get_out_channels(block):
    print(f"Inspecting block: {block}")
    print(f"Block type: {type(block)}")
    print(f"Block attributes: {dir(block)}")
    if hasattr(block, "conv"):
        return block.conv.out_channels
    if hasattr(block, "pointwise"):          # depthwise blocks
        return block.pointwise.out_channels
    raise ValueError("Cannot infer out_channels")

def get_in_channels(block):
    if hasattr(block, "conv"):
        return block.conv.in_channels
    if hasattr(block, "pointwise"):
        return block.pointwise.in_channels
    raise ValueError("Cannot infer in_channels")
