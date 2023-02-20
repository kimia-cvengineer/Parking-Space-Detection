import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''

    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        print("************** Focal loss **************")

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        print("In forward")
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        print("HERE")
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
