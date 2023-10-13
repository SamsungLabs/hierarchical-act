import torch

def reparametrize_normal(mu:torch.Tensor, logvar:torch.Tensor, 
                         training:bool =True)->torch.Tensor:
    if training:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        return mu