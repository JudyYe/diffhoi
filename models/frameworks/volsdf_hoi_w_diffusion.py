import functools
import os.path as osp
import copy
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d.transforms.rotation_conversions as rot_cvt
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.loss.chamfer import knn_points
from models.sd  import SDLoss

from models.articulation import get_artnet
from models.cameras.gt import PoseNet
from models.frameworks.volsdf import SingleRenderer, VolSDF, volume_render_flow
from utils import rend_util
from utils.logger import Logger
from .volsdf_hoi import VolSDFHoi, MeshRenderer
from .volsdf_hoi import Trainer as HoiTrainer
from models.blending import hard_rgb_blend, softmax_rgb_blend, volumetric_rgb_blend
from jutils import geom_utils, mesh_utils, model_utils, hand_utils





class Trainer(HoiTrainer):
    def __init__(self, model: VolSDFHoi, device_ids=[0], batched=True, args=None):
        super().__init__(model, device_ids, batched, args)

    def sample_jHand_camera(self, indices, model_input, ground_truth, H, W):
        return 

    def sample_jTc(self, indices, model_input, ground_truth):
        jTc, jTc_n, jTh, jTh_n = self.get_jTc(indices, model_input, ground_truth)
        # switch jTc with a sampled point on sphere
        rot_T = geom_utils.random_rotations(len(jTc), device=jTc.device).reshape(len(jTc), 3, 3)

        norm = mesh_utils.get_camera_dist(wTc=jTc)  # we want to get camera dist, and put it to -z? 
        zeros = torch.zeros_like(norm)
        trans = torch.stack([zeros, zeros, -norm], -1).unsqueeze(-1)

        jTc = geom_utils.rt_to_homo(rot_T, -rot_T@trans)
        # jTc_n = None
        return jTc, jTc_n, jTh, jTh_n 

    def forward(self, 
             args,
             indices,
             model_input,
             ground_truth,
             render_kwargs_train: dict,
             it: int):
        device = self.device
        N = len(indices)
        indices = indices.to(device)
        full_frame_iter = self.training and self.args.training.render_full_frame and it % 2 == 0
        # 1. GET POSES
        if full_frame_iter:
            jTc, jTc_n, jTh, jTh_n = self.sample_jTc(indices, model_input, ground_truth)
        else:
            jTc, jTc_n, jTh, jTh_n = self.get_jTc(indices, model_input, ground_truth)

        # NOTE: znear and zfar is important: distance of camera center to world origin
        cam_norm = jTc_n[..., 0:4, 3]
        cam_norm = cam_norm[..., 0:3] / cam_norm[..., 3:4]  # (N, 3)
        norm = torch.norm(cam_norm, dim=-1)
        
        render_kwargs_train['far'] = zfar = (norm + args.model.obj_bounding_radius).cpu().item()
        render_kwargs_train['near'] = znear = (norm - args.model.obj_bounding_radius).cpu().item()

        if full_frame_iter:
            render_kwargs_train_copy = deepcopy(render_kwargs_train)
            render_kwargs_train_copy['H'] = 64
            render_kwargs_train_copy['W'] = 64
            render_kwargs_train_copy['N_samples'] = 32  # low resolution smaple
            render_kwargs_train = render_kwargs_train_copy
        # else:
            # render_kwargs_train['H'] = self.H
            # render_kwargs_train['W'] = self.W
            # render_kwargs_train['N_samples'] = self.args.model.setdefault('N_samples', 128)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        intrinsics = self.focalnet(indices, model_input, ground_truth, H=H, W=W)
        intrinsics_n = self.focalnet(model_input['inds_n'].to(device), model_input, ground_truth, H=H,W=W)

        # 2. RENDER MY SCENE 
        # 2.1 RENDER HAND
        # hand FK
        hA = self.model.hA_net(indices, model_input, None)
        hA_n = self.model.hA_net(model_input['inds_n'].to(device), model_input, None)
        hHand, hJoints = self.hand_wrapper(None, hA, texture=self.model.uv_text)
        jHand = mesh_utils.apply_transform(hHand, jTh)
        jJoints = mesh_utils.apply_transform(hJoints, jTh)
        cJoints = mesh_utils.apply_transform(jJoints, geom_utils.inverse_rt(mat=jTc, return_mat=True))
        extras['hand'] = jHand
        extras['jJoints'] = jJoints
        extras['cJoints'] = cJoints

        hHand_n, hJoints_n = self.hand_wrapper(None, hA_n, texture=self.model.uv_text)
        jJoints_n = mesh_utils.apply_transform(hJoints_n, jTh_n)
        cJoints_n = mesh_utils.apply_transform(jJoints_n, geom_utils.inverse_rt(mat=jTc_n, return_mat=True))
        extras['jJoints_n'] = jJoints_n
        extras['cJoints_n'] = cJoints_n

        iHand = self.mesh_renderer(
            geom_utils.inverse_rt(mat=jTc, return_mat=True), intrinsics, jHand, **render_kwargs_train
        )
        extras['iHand'] = iHand


        # 2.2 RENDER OBJECT
        # volumetric rendering
        # rays in canonical object frame
        if self.training:
            if full_frame_iter:
                rays_o, rays_d, select_inds = rend_util.get_rays(
                    jTc, intrinsics, H, W, N_rays=-1)
                # print('render full frame', rays_o.shape)
            else:
                rays_o, rays_d, select_inds = rend_util.get_rays(
                    jTc, intrinsics, H, W, N_rays=args.data.N_rays)
                # print('render non full frame', rays_o.shape)
        else:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                jTc, intrinsics, H, W, N_rays=-1)
        
        rgb_obj, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)
        flow_12 = volume_render_flow(depth_v, select_inds, intrinsics, intrinsics_n, jTc, jTc_n, **render_kwargs_train)
        # rgb: (N, R, 3), mask/depth: (N, R, ), flow: (N, R, 2?)
        iObj = {'rgb': rgb_obj, 'depth': depth_v, 'mask': extras['mask_volume'], 'flow': flow_12}
        extras['iObj'] = iObj
        extras['select_inds'] = select_inds
        # 2.3 BLEND!!!
        # blended rgb, detph, mask, flow
        iHoi = self.blend(iHand, iObj, select_inds, znear, zfar, **args.blend_train)
        extras['iHoi'] = iHoi

        # 3. GET GROUND TRUTH SUPERVISION 
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))
        # [B, N_rays]
        target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds).float()
        # [B, N_rays]
        target_obj = torch.gather(model_input["obj_mask"].to(device), 1, select_inds).float()
        target_hand = torch.gather(model_input["hand_mask"].to(device), 1, select_inds).float()
        extras['target_rgb'] = target_rgb
        extras['target_mask'] = target_mask

        # masks for mask loss: # (N, P, 2)
        # GT is marked as hand not object, AND predicted object depth is behind
        ignore_obj = ((target_hand > 0) & ~(target_obj > 0)) & (iObj['depth'] > iHand['depth']) 
        ignore_hand = (~(target_hand > 0) & (target_obj > 0)) & (iObj['depth'] < iHand['depth'])
        label_target = torch.stack([target_hand, target_obj, torch.zeros_like(target_obj)], -1) # (N, P, 3)
        extras['label_target'] = label_target

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
        extras['implicit_nablas_norm'] = nablas_norm

        losses = OrderedDict()
        self.get_reg_loss(losses, extras)
        if full_frame_iter: 
            self.get_fullframe_reg_loss(losses, extras)
        else:
            self.get_reproj_loss(losses, extras, ground_truth, model_input, render_kwargs_train)
            self.get_temporal_loss(losses, extras)

        loss = 0
        for k, v in losses.items():
            if args.training.backward == 'pose' and 'contact' in k:
                continue
            loss += losses[k]
        
        losses['total'] = loss

        alpha, beta = self.model.forward_ab()
        alpha = alpha.data
        beta = beta.data
        extras['scalars'] = {'beta': beta, 'alpha': alpha}
        
        extras['obj_ignore'] = ignore_obj
        extras['hand_ignore'] = ignore_hand

        extras['obj_mask_target'] = model_input['obj_mask']
        extras['hand_mask_target'] = model_input['hand_mask']
        extras['mask_target'] = model_input['object_mask']
        extras['flow'] = iHoi['flow']
        extras['hA'] = hA
        extras['intrinsics'] = intrinsics

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
        
    def val(self, logger: Logger, ret, to_img_fn, it, render_kwargs_test):
        mesh_utils.dump_meshes(osp.join(logger.log_dir, 'hand_meshes/%08d' % it), ret['hand'])
        logger.add_meshes('hand',  osp.join('hand_meshes/%08d_0.obj' % it), it)
        
        print(ret['hand_mask_target'].device, ret['label_target'].device, ret['iHoi']['label'].device)
        # import pdb
        # pdb.set_trace()
        mask = torch.cat([
            to_img_fn(ret['hand_mask_target'].unsqueeze(-1).float()).repeat(1, 3, 1, 1),
            to_img_fn(ret['obj_mask_target'].unsqueeze(-1)).repeat(1, 3, 1, 1),
            to_img_fn(ret['mask_target'].unsqueeze(-1)).repeat(1, 3, 1, 1),
            to_img_fn(ret['label_target'].cpu()),
            ], -1)

        logger.add_imgs(mask, 'gt/hoi_mask_gt', it)

        mask = torch.cat([
            to_img_fn(ret['iHand']['mask'].unsqueeze(-1)).repeat(1, 3, 1, 1),
            to_img_fn(ret['iObj']['mask'].unsqueeze(-1)).repeat(1, 3, 1, 1),
            to_img_fn(ret['iHoi']['mask'].unsqueeze(-1)).repeat(1, 3, 1, 1),
            to_img_fn(ret['iHoi']['label']),
            ], -1)
        logger.add_imgs(mask, 'hoi/hoi_mask_pred', it)
        mask = torch.cat([
            to_img_fn(ret['hand_ignore'].unsqueeze(-1).float()),
            to_img_fn(ret['obj_ignore'].unsqueeze(-1).float()),
            ], -1)
        logger.add_imgs(mask, 'hoi/hoi_ignore_mask', it)

        image = torch.cat([
            to_img_fn(ret['iHand']['rgb']),
            to_img_fn(ret['iObj']['rgb']),
            to_img_fn(ret['iHoi']['rgb']),
        ], -1)
        logger.add_imgs(image, 'hoi/hoi_rgb_pred', it)

        depth_v_hand = ret['iHand']['depth'].unsqueeze(-1)
        depth_v_obj = ret['iObj']['depth'].unsqueeze(-1)
        depth_v_hoi = torch.minimum(depth_v_hand, depth_v_obj)
        hand_front = (depth_v_hand < depth_v_obj).float()
        depth = torch.cat([
            to_img_fn(depth_v_hand/(depth_v_hand.max()+1e-10)),
            to_img_fn(depth_v_obj/(depth_v_obj.max()+1e-10)),
            to_img_fn(depth_v_hoi/(depth_v_obj.max()+1e-10)),
            to_img_fn(hand_front)
        ], -1)
        logger.add_imgs(depth, 'hoi/hoi_depth_pred', it)

        # # vis depth in point cloud
        depth_hand = to_img_fn(depth_v_hand)
        depth_obj = to_img_fn(depth_v_obj)
        _, _, H, W = depth_hand.shape
        cameras = self.mesh_renderer.get_cameras(None, ret['intrinsics'], H, W)

        depth_hand = mesh_utils.depth_to_pc(depth_hand, cameras=cameras)
        depth_obj = mesh_utils.depth_to_pc(depth_obj, cameras=cameras)
        
        depth_hand = mesh_utils.pc_to_cubic_meshes(pc=depth_hand, eps=5e-2)
        depth_obj = mesh_utils.pc_to_cubic_meshes(pc=depth_obj, eps=5e-2)

        mesh_utils.dump_meshes(
            osp.join(logger.log_dir, 'meshes/%08d_hand' % it), 
            depth_hand)
        mesh_utils.dump_meshes(osp.join(logger.log_dir, 'meshes/%08d_obj' % it), 
            depth_obj)

        depth_hoi = mesh_utils.join_scene_w_labels([depth_hand, depth_obj], 3)
        image_list = mesh_utils.render_geom_rot(depth_hoi, 'circle', cameras=cameras, view_centric=True)
        logger.add_gifs(image_list, 'hoi/hoi_depth_pointcloud', it)
        # logger.add_meshes()
        # # depth_hand.textures = mesh_utils.pad_texture(depth_hand, 'yellow')
        # # depth_obj.textures = mesh_utils.pad_texture(depth_hand, 'white')
        # # # depth_hoi = mesh_utils.join_scene([depth_hand, depth_obj])


def get_model(args, data_size=-1, **kwargs):
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
    model_config['oTh_cfg'] = args.oTh    
    model_config['text_cfg'] = args.hand_text

    hA_cfg = {'key': 'hA', 'data_size': data_size, 'mode': args.hA.mode}
    model_config['hA_cfg'] = hA_cfg

    model = VolSDFHoi(**model_config)
    
    ## render_kwargs
    max_radius = kwargs.get('cam_norm', args.data.scale_radius)
    render_kwargs_train = {
        'near': max_radius - args.model.obj_bounding_radius, # args.data.near,
        'far':  max_radius + args.model.obj_bounding_radius, # args.data.far,
        'batched': True,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
        'max_upsample_steps': args.model.setdefault('max_upsample_iter', 5),
        'use_nerfplusplus': args.model.setdefault('outside_scene', 'builtin') == 'nerf++',
        'obj_bounding_radius': args.model.obj_bounding_radius,
        'N_samples': args.model.setdefault('N_samples', 128),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    render_kwargs_test['calc_normal'] = True

    devices = kwargs.get('device', args.device_ids)
    trainer = Trainer(model, devices, batched=render_kwargs_train['batched'], args=args)
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer, volume_render_flow
