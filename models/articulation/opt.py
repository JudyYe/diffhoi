import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from jutils import mesh_utils

from .gt import GTNet

def init_net_by_value(dataset, key, ):
    """return a (N, D) tensor"""
    value = []
    for i in range(len(dataset)):
        value.append(dataset[i][key])
    value = torch.stack(value, dim=0)
    return value


class ArtNet(nn.Module):
    def __init__(self, key='hA', data_size=1000, dim=45):
        super().__init__()
        self.key = key
        self.value = nn.Parameter(torch.zeros([data_size, 45]))
        self.base = GTNet(key)

    def init_net(self, dataset=None):
        self.base_value = init_net_by_value(dataset, self.key)
        self.value = nn.Parameter(torch.zeros_like(self.base_value))

    def forward(self, inds, model_input=None, gt=None):
        v = self.base(inds, model_input, gt)
        dv = self.value[inds] #torch.gather(self.value, 0, torch.stack([inds]*45, -1))
        
        return v + dv


def get_artnet(**kwargs):
    return ArtNet(**kwargs)