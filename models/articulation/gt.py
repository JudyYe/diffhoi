import torch.nn as nn


class GTNet(nn.Module):
    def __init__(self, key='hA', **kwargs):
        super().__init__()
        self.key = key

    def init_net(self, dataloader=None):
        return 

    def forward(self, inds, model_input, gt):
        device = inds.device
        N = len(inds)

        v_n = model_input['%s_n' % self.key].to(device) 
        v = model_input['%s' % self.key].to(device)

        m_nxt = (inds == model_input['inds_n'].to(device)).float()  # N, 
        
        ndims = inds.ndim
        m_nxt = m_nxt.view(N, *([1, ] * ndims))
        rtn = v_n * m_nxt + v * (1 - m_nxt)  
        return rtn


def get_artnet(**kwargs):
    return GTNet(**kwargs)