import torch
import torch.nn as nn
import torch.nn.functional as F
from jutils import geom_utils, mesh_utils


class PoseNet(nn.Module):
    def __init__(self, key='c2w', inverse=False):
        super().__init__()
        self.key = key
        self.inverse = inverse

    def forward(self, inds, model_input, gt, **kwargs):
        """returns extrinsic stored in model_input: c2w_n or c2w, 
        depending on if inds == model_input['inds_n']"""
        device = inds.device
        N = len(inds)
        # return  model_input['c2w'].to(device)
        c2w_n = model_input['%s_n' % self.key].to(device)  # (N, 4, 4)
        c2w = model_input['%s' % self.key].to(device)
        if self.inverse:
            c2w_n = geom_utils.inverse_rt(mat=c2w_n, return_mat=True)
            c2w = geom_utils.inverse_rt(mat=c2w, return_mat=True)
        m_nxt = (inds == model_input['inds_n'].to(device)).float()
        m_nxt = m_nxt.view(N, 1, 1)
        rtn = c2w_n * m_nxt + c2w * (1 - m_nxt)  # (N, 4, 4)
        return rtn


class FocalNet(nn.Module):
    def __init__(self, H=224, W=224, **kwargs):
        super().__init__()
    
    def forward(self, inds, model_input, gt, ndc=False, H=2, W=2, **kwargs):
        """Returns intrinsics in pixel/screen space with length H, W"""
        device = inds.device
        N = len(inds)
        intr = model_input["intrinsics"].to(device)
        intr_n = model_input["intrinsics_n"].to(device)

        m_nxt = (inds == model_input['inds_n'].to(device)).float()
        m_nxt = m_nxt.view(N, 1, 1)
        rtn = intr_n * m_nxt + intr * (1 - m_nxt)  # (N, 4, 4)
        if not ndc:
            rtn = mesh_utils.intr_from_ndc_to_screen(rtn, H, W)
        return rtn


def get_camera(args, **kwargs):
    posenet = PoseNet(**kwargs)
    focalnet = FocalNet(**kwargs)
    return posenet, focalnet