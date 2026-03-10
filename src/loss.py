
# loss functions for segmentation training

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        preds = F.softmax(preds, dim=1)

        num_classes = preds.shape[1]

        targets_onehot = F.one_hot(targets, num_classes)
        targets_onehot = targets_onehot.permute(0,3,1,2).float()

        intersection = (preds * targets_onehot).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):

        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)

        total_loss = ce_loss + dice_loss

        return total_loss