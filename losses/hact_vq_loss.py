import gin
import torch
from torch.nn import functional as F

from .base import BaseLoss


@gin.configurable()
class HactVqLoss(BaseLoss):
    def __init__(self, 
                    action_weight:float=1,
                    pad_weight:float=0.1,
                    z_vq_weight:float=1,
                    h_vq_weight:float=1,
                    h_weight:float=1,
                    z_weight:float=1,
                    time_cost_weight:float=1):
        self.action_weight = action_weight
        self.pad_weight = pad_weight
        self.z_vq_weight = z_vq_weight
        self.h_vq_weight = h_vq_weight
        self.h_weight = h_weight
        self.z_weight = z_weight
        self.time_cost_weight = time_cost_weight
        
        self.h_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.z_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.time_cost_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        #self.is_pad_bce_loss = torch.nn.BCELoss(reduction='none')
        self.is_pad_bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _compute_action_loss(self, actions, actions_hat, is_pad):
        all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        return l1

    def _compute_padding_loss(self, is_pad, is_pad_hat_logit):
        is_pad = is_pad.unsqueeze(-1).float()
        all_bce = self.is_pad_bce_loss(is_pad_hat_logit, is_pad)
        all_bce = all_bce.mean()
        return all_bce

    def _compute_vq_loss(self, z_e, z_q):
        loss_vq = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        return loss_vq + loss_commit

    def __call__(self, data, model_outputs, action_mean, action_std):
        actions = data['actions']
        is_pad = data['is_pad']
        actions_hat = model_outputs['pred_a']
        is_pad_hat_logit = model_outputs['pred_is_pad_logit']
        z_e = model_outputs['z_e']
        z_q = model_outputs['z_q']
        h_e = model_outputs['h_e']
        h_q = model_outputs['h_q']
        actions_norm = (actions-action_mean)/action_std
        actions_hat_norm = (actions_hat-action_mean)/action_std
        
        action_l1 = self._compute_action_loss(actions_norm, actions_hat_norm, is_pad)
        padding_bce = self._compute_padding_loss(is_pad, is_pad_hat_logit)
        z_vq = self._compute_vq_loss(z_e, z_q)
        h_vq = self._compute_vq_loss(h_e, h_q)

        low_level_total_loss = self.action_weight* action_l1 + self.pad_weight*padding_bce \
                        + self.z_vq_weight * z_vq + self.h_vq_weight * h_vq

        z_index = model_outputs['z_index'].flatten()
        h_index = model_outputs['h_index'].flatten()
        time_cost_index = model_outputs['time_cost_index'].flatten()
        
        pred_z_logit = model_outputs['pred_z_logit']
        pred_h_logit = model_outputs['pred_h_logit']
        pred_time_cost_logit = model_outputs['pred_time_cost_logit']
        
        z_ce =  self.z_ce_loss(pred_z_logit, z_index).mean()
        h_ce =  self.h_ce_loss(pred_h_logit, h_index).mean()
        time_cost_ce = self.time_cost_ce_loss(pred_time_cost_logit, time_cost_index).mean()
        high_level_total_loss = self.z_weight*z_ce + self.h_weight*h_ce + self.time_cost_weight*time_cost_ce
        
        total_loss = low_level_total_loss + high_level_total_loss
        losses = {
            'a_l1': action_l1,
            'padding': padding_bce,
            'z_vq': z_vq,
            'h_vq': h_vq,
            'z_ce': z_ce,
            'h_ce': h_ce,
            'time_cost_ce': time_cost_ce,
            'total_loss': total_loss
        }
        return losses