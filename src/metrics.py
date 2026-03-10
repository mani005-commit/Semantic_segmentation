
# dice score calculation

import torch
import torch.nn.functional as F


def dice_score(preds, targets, num_classes=21, smooth=1):

    preds = torch.argmax(preds, dim=1)

    preds_onehot = F.one_hot(preds, num_classes)
    targets_onehot = F.one_hot(targets, num_classes)

    preds_onehot = preds_onehot.permute(0,3,1,2).float()
    targets_onehot = targets_onehot.permute(0,3,1,2).float()

    intersection = (preds_onehot * targets_onehot).sum(dim=(2,3))
    union = preds_onehot.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))

    dice = (2 * intersection + smooth) / (union + smooth)

    dice = dice.mean()

    return dice.item()