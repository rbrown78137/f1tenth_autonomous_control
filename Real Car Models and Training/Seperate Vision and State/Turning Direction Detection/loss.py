import torch
import torch.nn as nn

class AvailableLaneLoss(nn.Module):

    def __init__(self):

        super(AvailableLaneLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions, target):
        left_loss = self.bce(
            torch.sigmoid(predictions[..., 0:1]),
            target[..., 0:1],
        )
        right_loss = self.bce(
            torch.sigmoid(predictions[..., 1:2]),
            target[..., 1:2],
        )
        loss = (
            1 * left_loss + 1 * right_loss
        )
        return loss