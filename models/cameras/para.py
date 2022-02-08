import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnutils import geom_utils, mesh_utils


"""from nerfmm: https://github.com/JudyYe/nerfmm/tree/dfa552bf4c2967d11dcd2ea8462fda2cbc96c4df/models"""

class PoseNet(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None, init_dist=2):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super().__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is None:
            # identity rotation +  (0, 0, 2)
            zeros = torch.zeros([num_cams, 3, ], dtype=torch.float32)
            twos =  torch.zeros([num_cams, 3, ], dtype=torch.float32)
            twos[..., -1] = -init_dist
            # NOTE: inverse?????? 
            init_c2w = geom_utils.axis_angle_t_to_matrix(zeros, twos)  
        self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id, model_input, gt):        
        r = torch.gather(self.r, 0, torch.stack(3*[cam_id], -1))  # (3, ) axis-angle
        t = torch.gather(self.t, 0, torch.stack(3*[cam_id], -1))  # (3, )
        
        # rt as a correction to init_c2w
        c2w = geom_utils.axis_angle_t_to_matrix(r, t)
        
        # learn a delta pose between init pose and target pose, if a init pose is provided
        c2w = c2w @ self.init_c2w[cam_id]  #  torch.gather(self.init_c2w, 0, cam_id)
        return c2w


class FocalNet(nn.Module):
    def __init__(self, num_cams, H, W, learn_f, learn_pp, fx_only, order=2, init_focal=None, init_px=0, init_py=0):
        super().__init__()
        self.num_cams = num_cams
        # self.H = H
        # self.W = W
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.
        
        zeros = torch.zeros([num_cams, 2], dtype=torch.float32)
        # final focal length = H/W * init_f_ndc * coe_x**2
        self.init_f_ndc = nn.Parameter(init_focal + zeros, requires_grad=False)
        # final pp = HW/2 * (init_pp_ndc + pp)
        self.init_pp_ndc =  nn.Parameter(0.5+zeros, requires_grad=False)
        
        if self.fx_only:
            coe_x = torch.ones([num_cams, 1], dtype=torch.float32, requires_grad=False)
            self.fx = nn.Parameter(coe_x, requires_grad=learn_f)  # (1, )
        else:
            coe_x = torch.ones([num_cams, 1], dtype=torch.float32, requires_grad=False)
            coe_y = torch.ones([num_cams, 1], dtype=torch.float32, requires_grad=False)
            self.fx = nn.Parameter(coe_x, requires_grad=learn_f)  # (1, )
            self.fy = nn.Parameter(coe_y, requires_grad=learn_f)  # (1, )
        
        self.pp = nn.Parameter(torch.zeros([num_cams, 2]), requires_grad=learn_pp)

    def _get_focal(self, i, H, W):  # the i=None is just to enable multi-gpu training
        # final focal length = H/W * init_f_ndc * coe_x**2
        init_fx, init_fy = self.init_f_ndc[i].split(1, -1)

        if self.fx_only:
            if self.order == 2:
                fxfy = torch.cat([self.fx[i] ** 2 * W * init_fx, 
                                    self.fx[i] ** 2 * H * init_fy], -1)
            else:
                fxfy = torch.cat([self.fx[i] * W * init_fx, 
                                    self.fx[i] * H * init_fy], -1)
        else:
            if self.order == 2:
                fxfy = torch.cat([self.fx[i]**2 * W * init_fx, 
                                    self.fy[i]**2 * H * init_fy], -1)
            else:
                fxfy = torch.cat([self.fx[i] * W * init_fx, 
                                    self.fy[i] * H * init_fy], -1)
        return fxfy

    def _get_pp(self, i, H, W):  # the i=None is just to enable multi-gpu training
        # final pp = HW/2 * (init_pp_ndc + pp)
        px, py = self.pp[i].split(1, dim=-1)  # (N, 2)
        init_px, init_py = self.init_pp_ndc[i].split(1, -1)
        px = W * (init_px + px)
        py = H * (init_py + py)
        pxpy = torch.cat([px, py], -1)
        return pxpy
        
    def forward(self, i=None, model_input=None, gt=None, H=1, W=1, **kwargs):  # the i=None is just to enable multi-gpu training
        """[summary]

        Args:
            i ([type], optional): [description]. Defaults to None.
        Return:
            (N, 4, 4)
        """
        fxfy = self._get_focal(i, H, W)
        pxpy = self._get_pp(i, H, W)
        # print(fxfy, pxpy, fxfy.size(), pxpy.size())
        intrinsics = mesh_utils.get_k_from_fp(fxfy, pxpy)
        return intrinsics


def get_camera(args, datasize, **kwargs):
    camera_conf = {
        'num_cams': datasize,
        'learn_R': bool(args.camera.learn_R),
        'learn_t': bool(args.camera.learn_t),
        'init_dist': args.camera.init_dist,
    }
    posenet = PoseNet(**camera_conf)
    intr_conf = {
        'num_cams': datasize,
        'H': kwargs.get('H'),
        'W': kwargs.get('W'),
        'learn_f': bool(args.camera.learn_f),
        'learn_pp': bool(args.camera.learn_pp),
        'fx_only': bool(args.camera.fx_only),
        'order': args.camera.order_f,
        'init_focal': args.camera.init_f,
    }
    focalnet = FocalNet(**intr_conf)
    return posenet, focalnet


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    from utils import io_util
    from tools.vis_camera import visualize
    from models.cameras import get_camera as get_camera_factory
    torch.manual_seed(123)
    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)

    from dataio.DTU import SceneDataset

    train_dataset = SceneDataset(
        train_cameras=False,
        data_dir='./data/DTU/scan65')
    dataloader = DataLoader(train_dataset,
            batch_size=4,
            shuffle=True,
            pin_memory=config.data.get('pin_memory', False))
    (indices, model_input, ground_truth) = next(iter(dataloader))
    
    c2w = torch.stack(train_dataset.c2w_all).data.cpu().numpy()

    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = model_input['intrinsics'].data.cpu().numpy()[0]

    visualize(camera_matrix, extrinsics, 'logs/debug_cam/all.png')

    for mode in ['para', 'gt']:
        config.camera.mode = mode
        posenet, focal_net = get_camera_factory(config, 
            datasize=len(train_dataset), H=train_dataset.H, W=train_dataset.W)

        c2w = posenet(indices, model_input, ground_truth).cpu().detach().numpy()
        camera_matrix = focal_net(indices, model_input, ground_truth,
            H=train_dataset.H,W=train_dataset.W).cpu().detach().numpy()[0]
    
        extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
        visualize(camera_matrix, extrinsics, 'logs/debug_cam/%s.png' % mode)

        print(mode, camera_matrix, extrinsics)

    