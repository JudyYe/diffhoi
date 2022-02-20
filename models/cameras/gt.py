import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inds, model_input, gt):
        device = inds.device
        N = len(inds)
        # return  model_input['c2w'].to(device)
        c2w_n = model_input['c2w_n'].to(device)  # (N, 4, 4)
        c2w = model_input['c2w'].to(device)
        m_nxt = (inds == model_input['inds_n'].to(device)).float()
        m_nxt = m_nxt.view(N, 1, 1)
        rtn = c2w_n * m_nxt + c2w * (1 - m_nxt)  # (N, 4, 4)
        return rtn


class FocalNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inds, model_input, gt, **kwargs):
        device = inds.device
        N = len(inds)
        intr = model_input["intrinsics"].to(device)
        intr_n = model_input["intrinsics_n"].to(device)

        m_nxt = (inds == model_input['inds_n'].to(device)).float()
        m_nxt = m_nxt.view(N, 1, 1)
        rtn = intr_n * m_nxt + intr * (1 - m_nxt)  # (N, 4, 4)
        return rtn


def get_camera(args, **kwargs):
    posenet = PoseNet()
    focalnet = FocalNet()
    return posenet, focalnet