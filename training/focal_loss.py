import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for highly imbalanced binary classification tasks.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Weighting factor for the positive class (default: 0.25).
        gamma (float): Focusing parameter for hard examples (default: 2.0).
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unscaled model outputs [batch_size, 1] or [batch_size]
            targets: Binary labels (0 or 1) of same shape
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t: probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t: class weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # focal modulation weight
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
