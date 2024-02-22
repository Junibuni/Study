import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                pass
            elif isinstance(self.alpha, np.ndarray):
                alpha = torch.from_numpy(self.alpha)
                alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
            elif isinstance(alpha, torch.Tensor):
                alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)   

            alpha = self.alpha[targets.data.view(-1)]

            focal_loss = focal_loss * alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss