
# computing FLOPs of segmentation model

import torch
from fvcore.nn import FlopCountAnalysis

from model import UNet


def compute_flops():

    # create model
    model = UNet(num_classes=21)

    model.eval()

    # dummy input
    dummy_input = torch.randn(1, 3, 300, 300)

    # compute FLOPs
    flops = FlopCountAnalysis(model, dummy_input)

    total_flops = flops.total()

    print("Total FLOPs:", total_flops)

    print("FLOPs (GFLOPs):", total_flops / 1e9)


if __name__ == "__main__":

    compute_flops()