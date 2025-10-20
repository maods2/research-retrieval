import torch
from torch import nn
from typing import Dict, Any


class SupervisedHashingLoss(nn.Module):
    """
    Supervised Hashing Loss for Deep Supervised Hashing (DSH).
    
    This loss function implements the triplet-like loss used in supervised hashing,
    which consists of three components:
    1. Positive pair loss: minimize distance between similar samples
    2. Negative pair loss: maximize distance between dissimilar samples (with margin)
    3. Regularization: encourage binary-like outputs
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.margin = config.get('margin', 5.0)
        self.alpha = config.get('alpha', 0.01)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def forward(self, x_out: torch.Tensor, y_out: torch.Tensor, target_equals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the supervised hashing loss.
        
        Args:
            x_out: Hash codes for first image in pair [batch_size, code_size]
            y_out: Hash codes for second image in pair [batch_size, code_size]
            target_equals: Binary tensor indicating if pairs are similar (0) or dissimilar (1)
            
        Returns:
            Dictionary containing total loss and component losses
        """
        # Compute squared distance between hash codes
        squared_loss = torch.mean(self.mse_loss(x_out, y_out), dim=1)
        
        # T1: Positive pair loss - minimize distance for similar pairs
        # target_equals = 0 for similar pairs, 1 for dissimilar pairs
        positive_pair_loss = 0.5 * (1 - target_equals) * squared_loss
        mean_positive_pair_loss = torch.mean(positive_pair_loss)
        
        # T2: Negative pair loss - maximize distance for dissimilar pairs (with margin)
        zeros = torch.zeros_like(squared_loss)
        margin = self.margin * torch.ones_like(squared_loss)
        negative_pair_loss = 0.5 * target_equals * torch.max(zeros, margin - squared_loss)
        mean_negative_pair_loss = torch.mean(negative_pair_loss)
        
        # T3: Regularization - encourage binary-like outputs
        mean_value_regularization = self.alpha * (
            self.l1_loss(torch.abs(x_out), torch.ones_like(x_out)) +
            self.l1_loss(torch.abs(y_out), torch.ones_like(y_out))
        )
        
        # Total loss
        total_loss = mean_positive_pair_loss + mean_negative_pair_loss + mean_value_regularization
        
        return {
            'total_loss': total_loss,
            'positive_loss': mean_positive_pair_loss,
            'negative_loss': mean_negative_pair_loss,
            'regularization_loss': mean_value_regularization
        }
