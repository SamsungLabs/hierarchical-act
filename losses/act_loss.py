import gin
import torch
from torch.nn import functional as F

from .base import BaseLoss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld

@gin.configurable()
class ActLoss(BaseLoss):
    def __init__(self, 
                    kl_weight:float=10.):
        self.kl_weight = kl_weight

    def _compute_action_loss(self, actions, actions_hat, is_pad):
        all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        return l1

    def _compute_KL_loss(self, mu, logvar):
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        return total_kld[0]

    def __call__(self, data, model_outputs, action_mean, action_std):
        actions = data['actions']
        is_pad = data['is_pad']
        actions_hat = model_outputs['a_hat']
        mu = model_outputs['mu']
        logvar = model_outputs['logvar']

        actions_norm = (actions-action_mean)/action_std
        actions_hat_norm = (actions_hat-action_mean)/action_std
        
        l1 = self._compute_action_loss(actions_norm, actions_hat_norm, is_pad)
        kl = self._compute_KL_loss(mu, logvar)
        total_loss = l1 + self.kl_weight*kl

        losses = {
            'l1': l1,
            'kl': kl,
            'total_loss': total_loss
        }
        return losses