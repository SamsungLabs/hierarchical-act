import gin
import torch

@gin.configurable('Optimizer')
class TorchOptimizerBuilder():
    def __init__(self, name: str='', **kwargs):
        self.name = name
        self.kwargs = kwargs

    def build(self, params):
        if self.name == 'adam':
            optimizer_cls=torch.optim.Adam
        elif self.name == 'adamw':
            optimizer_cls=torch.optim.AdamW
        else:
            raise Exception("Not supported optimizer")
        return optimizer_cls(params, **self.kwargs)