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

        alpha = self.alpha[targets.data.view(-1)]

        focal_loss = focal_loss * alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MeanIoU(nn.Module):
    def __init__(self, num_classes):
        super(MeanIoU, self).__init__()
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)
        
    def reset(self):
        self.intersection.zero_()
        self.union.zero_()

    def update(self, y_true, y_pred):
        for cls in range(self.num_classes):
            intersection = torch.sum((y_true == cls) & (y_pred == cls))
            union = torch.sum((y_true == cls) | (y_pred == cls))
            self.intersection[cls] += intersection.item()
            self.union[cls] += union.item()

    def compute_iou(self):
        class_iou = self.intersection / self.union
        return class_iou

    def forward(self, y_true, y_pred):
        self.reset()
        self.update(y_true, y_pred)
        class_iou = self.compute_iou()
        mean_iou = torch.mean(class_iou)
        return mean_iou