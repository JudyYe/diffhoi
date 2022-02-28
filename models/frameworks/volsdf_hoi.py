from sys import prefix

import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d.transforms.rotation_conversions as rot_cvt

from models.base import ImplicitSurface, NeRF, RadianceNet
from models.frameworks.volsdf import SingleRenderer, VolSDF, volume_render_flow
from utils import io_util, train_util, rend_util
from utils.dist_util import is_master
from utils.logger import Logger

from jutils import geom_utils, mesh_utils


class RelPoseNet(nn.Module):
    """ Per-frame R,t correction @ base s,R,t
    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_frames, learn_R, learn_t, 
                 init_pose, learn_base_s, learn_base_R, learn_base_t):
        """
        Args:
            num_frames (_type_): N
            learn_R (_type_): True/False
            learn_T (_type_): True/False
            init_pose (_type_): base pose srt initialziation (1, 4, 4)
            learn_base_s (_type_): True/False
            learn_base_R (_type_): True/False
            learn_base_t (_type_): True/False
        """
        super().__init__()
        if init_pose is None:
            init_pose = torch.eye(4).unsqueeze(0)
        base_r, base_t, base_s = geom_utils.homo_to_rt(init_pose)
        # use axisang
        self.base_r = nn.Parameter(geom_utils.matrix_to_axis_angle(base_r), learn_base_R)
        self.base_s = nn.Parameter(base_s, learn_base_s)
        self.base_t = nn.Parameter(base_t, learn_base_t)

        self.r = nn.Parameter(torch.zeros(size=(num_frames, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_frames, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id, *args, **kwargs):
        r = torch.gather(self.r, 0, torch.stack(3*[cam_id], -1))  # (3, ) axis-angle
        t = torch.gather(self.t, 0, torch.stack(3*[cam_id], -1))  # (3, )
        frameTbase = geom_utils.axis_angle_t_to_matrix(r, t)

        N = len(r)

        base = geom_utils.rt_to_homo(
            rot_cvt.axis_angle_to_matrix(self.base_r), self.base_t, self.base_s)
        base = base.repeat(N, 1, 1)

        frame_pose = frameTbase @ base
        return frame_pose


class VolSDFHoi(VolSDF):
    def __init__(self,
                 data_size=100, 
                 beta_init=0.1,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 obj_bounding_radius=3.0,
                 use_nerfplusplus=False,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__(beta_init, speed_factor, 
            input_ch, W_geo_feat, obj_bounding_radius, use_nerfplusplus, 
            surface_cfg, radiance_cfg)
        # TODO: initialize pose
        self.oTh = RelPoseNet(data_size, learn_R=True, learn_t=True, 
            init_pose=None, learn_base_s=True, learn_base_R=True, learn_base_t=True)

class MeshRenderer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, cTw, pix_intr, meshes:Meshes, H, W):
        K = mesh_utils.intr_from_screen_to_ndc(pix_intr, H, W)
        f, p = mesh_utils.get_fxfy_pxpy(K)
        rot, t, _ = geom_utils.homo_to_rt(cTw)
        cameras = PerspectiveCameras(f, p, R=rot.transpose(-1, -2), T=t)

        image = mesh_utils.render_mesh(meshes, cameras, rgb_mode=True, depth_mode=True)
        return image

# class HandMeshRenderer(nn.Module):

class Trainer(nn.Module):
    def __init__(self, model: VolSDFHoi, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        self.mesh_renderer = MeshRenderer()
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]

        self.posenet: nn.Module = None
        self.focalnet: nn.Module = None

    def init_camera(self, posenet, focalnet):
        self.posenet = posenet
        self.focalnet = focalnet

    def forward(self, 
             args,
             indices,
             model_input,
             ground_truth,
             render_kwargs_train: dict,
             it: int):
        device = self.device
        indices = indices.to(device)
        # palmTcam
        hTc = self.posenet(indices, model_input, ground_truth)
        hTc_n = self.posenet(model_input['inds_n'].to(device), model_input, ground_truth)
        
        # apply object to hand pose, include scale!, R, t
        oTh = self.model.oTh(indices)
        oTh_n = self.model.oTh(model_input['inds_n'].to(device))
        oTc = oTh @ hTc
        oTc_n = oTh_n @ hTc_n
        
        c2w = oTc
        c2w_n = oTc_n

        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        intrinsics = self.focalnet(indices, model_input, ground_truth, H=H, W=W)
        intrinsics_n = self.focalnet(model_input['inds_n'].to(device), model_input, ground_truth, H=H,W=W)

        # mesh rendering for hand
        hHand = model_input['hand']
        # rtn = self.hand_wrapper.to_palm(
        #     model_input['hRot'].cuda(), model_input['hA'].cuda(), return_mesh=False)
        # palmHand = rtn[0]

        # camTpalm = model_input['camTpalm'] jgeom_utils.axis_angle_t_to_matrix(
            # t=mesh_utils.weak_to_full_persp(scale, ppoint, self.predcam_st[..., :1], self.predcam_st[..., 1:]))

        iHand = self.mesh_renderer(
            geom_utils.inverse_rt(mat=hTc, return_mat=True), 
            intrinsics, hHand, H, W)

        # volumetric rendering

        # rays in canonical object frame
        if self.training:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                c2w, intrinsics, H, W, N_rays=args.data.N_rays)
        else:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                c2w, intrinsics, H, W, N_rays=-1)

        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))
        # [B, N_rays]
        target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds).float()
        # [B, N_rays, 2]
        target_flow_fw = torch.gather(model_input['flow_fw'].to(device), 1, torch.stack(2*[select_inds], -1))

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        elif args.training.fg == 1:
            mask_ignore = target_mask
        else:
            mask_ignore = None
        
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)
        flow_12 = volume_render_flow(depth_v, select_inds, intrinsics, intrinsics_n, c2w, c2w_n, **render_kwargs_train)
        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        
        # [B, N_rays, ]
        #---------- OPTION1: just flatten and use all nablas
        # nablas = nablas.flatten(-3, -2)
        
        #---------- OPTION2: using only one point each ray: this may be what the paper suggests.
        # @ VolSDF section 3.5, "combine a SINGLE random uniform space point and a SINGLE point from \mathcal{S} for each pixel"
        _, _ind = extras['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        
        eik_bounding_box = args.model.obj_bounding_radius
        eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(device)
        _, nablas_eik, _ = self.model.implicit_surface.forward_with_nablas(eikonal_points)
        nablas = torch.cat([nablas, nablas_eik], dim=-2)

        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)

        losses = OrderedDict()

        losses['loss_img'] = F.l1_loss(rgb, target_rgb, reduction='none')
        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

        losses['loss_mask'] = args.training.w_mask * F.l1_loss(extras['mask_volume'], target_mask, reduction='mean')
        # convert mask from screen space to NDC space -- better bound the flow?? 
        # [0, H] -1, 1
        max_H = max(render_kwargs_train['H'], render_kwargs_train['W'])
        losses['loss_fl_fw'] = (2/max_H)**2 * args.training.w_flow * F.mse_loss(flow_12, target_flow_fw.detach(), reduction='none')
        if mask_ignore is not None:
            losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
            losses['loss_fl_fw'] = (losses['loss_fl_fw'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10) 
        else:
            losses['loss_img'] = losses['loss_img'].mean()
            losses['loss_fl_fw'] = losses['loss_fl_fw'].mean()
        
        loss = 0
        for k, v in losses.items():
            loss += losses[k]
        
        losses['total'] = loss
        
        extras['implicit_nablas_norm'] = nablas_norm

        alpha, beta = self.model.forward_ab()
        alpha = alpha.data
        beta = beta.data
        extras['scalars'] = {'beta': beta, 'alpha': alpha}
        extras['select_inds'] = select_inds
        
        extras['hand_rgb'] = iHand['image']
        extras['flow'] = flow_12

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
        

    def val(self, logger: Logger, ret, to_img_fn, it, render_kwargs_test):
        # log hand
        logger.add_imgs(ret['hand_rgb'], 'val/predicted_hand', it)

        #----------- plot beta heat map
        beta_heat_map = to_img_fn(ret['beta_map']).permute(0, 2, 3, 1).data.cpu().numpy()
        beta_heat_map = io_util.gallery(beta_heat_map, int(np.sqrt(beta_heat_map.shape[0])))
        _, beta = self.model.forward_ab()
        beta = beta.data.cpu().numpy().item()
        # beta_min = beta_heat_map.min()
        beta_max = beta_heat_map.max().item()
        if beta_max != beta:
            ticks = np.linspace(beta, beta_max, 10).tolist()
        else:
            ticks = [beta]
        tick_labels = ["{:.4f}".format(b) for b in ticks]
        tick_labels[0] = "beta={:.4f}".format(beta)
        
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(beta_heat_map, vmin=beta, vmax=beta_max)
        cbar = fig.colorbar(ax_im, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        logger.add_figure(fig, 'val/beta_heat_map', it)
        
        #----------- plot iteration used for each ray
        max_iter = render_kwargs_test['max_upsample_steps']
        iter_usage_map = to_img_fn(ret['iter_usage'].unsqueeze(-1)).permute(0, 2, 3, 1).data.cpu().numpy()
        iter_usage_map = io_util.gallery(iter_usage_map, int(np.sqrt(iter_usage_map.shape[0])))
        iter_usage_map[iter_usage_map==-1] = max_iter+1
        
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(iter_usage_map, vmin=0, vmax=max_iter+1)
        ticks = list(range(max_iter+2))
        tick_labels = ["{:d}".format(b) for b in ticks]
        tick_labels[-1] = 'not converged'
        cbar = fig.colorbar(ax_im, ticks=ticks)
        cbar.ax.set_yticklabels(tick_labels)
        logger.add_figure(fig, 'val/upsample_iters', it)

def get_model(args, data_size=-1):
    model_config = {
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius,
        'W_geo_feat': args.model.setdefault('W_geometry_feature', 256),
        'speed_factor': args.training.setdefault('speed_factor', 1.0),
        'beta_init': args.training.setdefault('beta_init', 0.1)
    }
    
    surface_cfg = {
        'use_siren': args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init':  args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }
        
    radiance_cfg = {
        'use_siren': args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }
    model_config['data_size'] = data_size
    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg

    model = VolSDFHoi(**model_config)
    
    ## render_kwargs
    render_kwargs_train = {
        'near': args.data.near,
        'far': args.data.far,
        'batched': True,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
        'max_upsample_steps': args.model.setdefault('max_upsample_iter', 5),
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    render_kwargs_test['calc_normal'] = True
    
    trainer = Trainer(model, args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer, volume_render_flow
