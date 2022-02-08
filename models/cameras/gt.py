import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inds, model_input, gt):
        device = inds.device
        return  model_input['c2w'].to(device)


class FocalNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inds, model_input, gt, **kwargs):
        device = inds.device
        return model_input["intrinsics"].to(device)


def get_camera(args, **kwargs):
    posenet = PoseNet()
    focalnet = FocalNet()
    return posenet, focalnet