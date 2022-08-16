from __future__ import division

import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduce=False, label_smoothing = 0.1)

    def forward(self, pred, target, mask):
        loss = self.celoss(pred, target)
        return (loss*mask).sum() / mask.sum()

