from torch import nn

from loss.focal_loss import FocalLoss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)
        loss = self.wd * dice_loss + self.wb * focal_loss
        return loss
