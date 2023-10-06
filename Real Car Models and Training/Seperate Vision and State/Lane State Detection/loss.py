import torch
import torch.nn as nn

class LaneStateLoss(nn.Module):

    def __init__(self):

        super(LaneStateLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions, target):
        loss = self.bce(
            torch.sigmoid(predictions),
            target,
        )
        return loss