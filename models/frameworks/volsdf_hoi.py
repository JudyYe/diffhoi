import functools
import os.path as osp
import copy
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d.transforms.rotation_conversions as rot_cvt
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.loss.chamfer import knn_points
from pytorch3d.transforms import Transform3d
from models.sd  import SDLoss

from models.articulation import get_artnet
from models.cameras.gt import PoseNet
from models.frameworks.volsdf import SingleRenderer, VolSDF, volume_render_flow
from utils import rend_util
from utils.logger import Logger

from models.blending import hard_rgb_blend, softmax_rgb_blend, volumetric_rgb_blend
from jutils import geom_utils, mesh_utils, model_utils, hand_utils, plot_utils, image_utils



class RelPoseNet(nn.Module):
    """ Per-frame R,t correction @ base s,R,t
    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_frames, learn_R, learn_t, 
                 init_pose, learn_base_s, learn_base_R, learn_base_t, **kwargs):
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
        self.base_s = nn.Parameter(base_s[..., 0:1], learn_base_s)
        self.base_t = nn.Parameter(base_t, learn_base_t)

        self.r = nn.Parameter(torch.zeros(size=(num_frames, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_frames, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id, *args, **kwargs):
        r = torch.gather(self.r, 0, torch.stack(3*[cam_id], -1))  # (3, ) axis-angle
        t = torch.gather(self.t, 0, torch.stack(3*[cam_id], -1))  # (3, )
        frameTbase = geom_utils.axis_angle_t_to_matrix(r, t)

        N = len(r)

        base = geom_utils.rt_to_homo(
            rot_cvt.axis_angle_to_matrix(self.base_r), self.base_t, self.base_s.repeat(1, 3))
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
                 radiance_cfg=dict(),
                 oTh_cfg=dict(),
                 text_cfg=dict(),
                 hA_cfg=dict()):
        super().__init__(beta_init, speed_factor, 
            input_ch, W_geo_feat, obj_bounding_radius, use_nerfplusplus, 
            surface_cfg, radiance_cfg)
        # TODO: initialize pose
        if oTh_cfg['mode'] == 'learn':
            self.oTh = RelPoseNet(data_size, init_pose=None, **oTh_cfg)
        elif oTh_cfg['mode'] == 'gt':
            self.oTh = PoseNet('hTo', True)
        else:
            raise NotImplementedError('Not implemented oTh.model: %s' % oTh_cfg['mode'])
        # initialize uv texture of hand
        t_size = text_cfg.get('size', 4)
        uv_text = torch.ones([1, t_size, t_size, 3]); uv_text[..., 2] = 0
        self.uv_text = nn.Parameter(uv_text)
        self.uv_text_init = False

        # hand articulation
        hA_mode = hA_cfg.pop('mode')
        self.hA_net = get_artnet(hA_mode, hA_cfg)


class MeshRenderer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def get_cameras(self, cTw, pix_intr, H, W):
        K = mesh_utils.intr_from_screen_to_ndc(pix_intr, H, W)
        f, p = mesh_utils.get_fxfy_pxpy(K)
        if cTw is None:
            cameras = PerspectiveCameras(f, p,  device=pix_intr.device)
        else:
            rot, t, _ = geom_utils.homo_to_rt(cTw)
            # cMesh 
            cameras = PerspectiveCameras(f, p, R=rot.transpose(-1, -2), T=t, device=cTw.device)
        return cameras

    def forward(self, cTw, pix_intr, meshes:Meshes, H, W, far, **kwargs):
        cameras = self.get_cameras(None, pix_intr, H, W)

        # apply cameraTworld outside of rendering to support scaling.
        cMeshes = mesh_utils.apply_transform(meshes, cTw)  # to allow scaling                 
        image = mesh_utils.render_soft(cMeshes, cameras, 
            rgb_mode=True, depth_mode=True, normal_mode=True, xy_mode=True, 
            out_size=max(H, W))
        # image = mesh_utils.render_mesh(cMeshes, cameras, rgb_mode=True, depth_mode=True, out_size=max(H, W))
        # apply cameraTworld for depth 
        # depth is in camera-view but is in another metric! 
        # convert the depth unit to the normalized unit.
        with torch.no_grad():
            _, _, cTw_scale = geom_utils.homo_to_rt(cTw)
            wTc_scale = 1 / cTw_scale.mean(-1)
        image['depth'] = wTc_scale * image['depth']
        image['depth'] = image['mask'] * image['depth'] + (1 - image['mask']) * far
        return image


class Trainer(nn.Module):
    @staticmethod
    def lin2img(x):
        H = int(x.shape[1]**0.5)
        return rearrange(x, 'n (h w) c -> n c h w', h=H, w=H)

    def __init__(self, model: VolSDFHoi, device_ids=[0], batched=True, args=None):
        super().__init__()
        self.joint_frame = args.model.joint_frame
        self.obj_radius = args.model.obj_bounding_radius
        self.args = args
        self.H : int = 224 # training reso
        self.W : int = 224 # training reso

        self.model = model
        self.renderer = SingleRenderer(model)
        self.mesh_renderer = MeshRenderer()
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]

        self.posenet: nn.Module = None
        self.focalnet: nn.Module = None
        self.hand_wrapper = hand_utils.ManopthWrapper()

        # if args.training.w_diffuse > 0:
        self.sd_loss = SDLoss(args.novel_view.diffuse_ckpt, args, **args.novel_view.sd_para)
        self.sd_loss.init_model(self.device)

    def init_hand_texture(self, dataloader):
        color_list = []
        with torch.no_grad():
            for (indices, model_input, ground_truth) in dataloader:
                hand_color = (ground_truth['rgb'] * model_input['hand_mask'][..., None]).sum(1) / model_input['hand_mask'][..., None].sum(1)
                # (N, 3)
                color_list.append(hand_color)
            color_list = torch.cat(color_list, 0).mean(0).reshape(1, 1, 1, 3)  # 3
        t_size = self.args.hand_text.size
        uv_text = torch.zeros([1, t_size, t_size, 3]) + color_list.cpu()
        self.model.uv_text = nn.Parameter(uv_text.data)

    def init_camera(self, posenet, focalnet):
        self.posenet = posenet
        self.focalnet = focalnet

    def sample_jHand_camera(self, indices=None, model_input=None, ground_truth=None, H=64, W=64):
        jHand, jTc, jTh, intrinsics = self.get_jHand_camera(indices, model_input, ground_truth, H, W)
        jTc = self.sample_jTc_like(jTc, )
        # jTc = mesh_utils.sample_camera_extr_like(wTc=jTc)
        # when we sample intrincisc, we set pp to center 
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        return jHand, jTc, jTh, intrinsics

    def get_jHand_camera(self, indices, model_input, ground_truth, H, W):
        jTc, jTc_n, jTh, jTh_n = self.get_jTc(indices, model_input, ground_truth)
        intrinsics = self.focalnet(indices, model_input, ground_truth, H=H, W=W)
        
        hA = self.model.hA_net(indices, model_input, None)
        # hand FK
        hHand, _ = self.hand_wrapper(None, hA, texture=self.model.uv_text)
        jHand = mesh_utils.apply_transform(hHand, jTh)

        return jHand, jTc, jTh, intrinsics

    def render(self, jHand, jTc, intrinsics, render_kwargs, use_surface_render='sphere_tracing', blend='hard'):
        N = 1
        H, W = render_kwargs['H'], render_kwargs['W']

        norm = mesh_utils.get_camera_dist(wTc=jTc)
        render_kwargs['far'] = zfar = (norm + self.obj_radius).cpu().item()
        render_kwargs['near'] = znear = (norm - self.obj_radius).cpu().item()
        
        model = self.model
        if use_surface_render:
            assert use_surface_render == 'sphere_tracing' or use_surface_render == 'root_finding'
            from models.ray_casting import surface_render
            render_fn = functools.partial(surface_render, model=model, ray_casting_algo=use_surface_render, 
                ray_casting_cfgs={'near': render_kwargs['near'], 'far': render_kwargs['far']})
        else:
            render_fn = self.renderer
        
        # mesh rendering 
        iHand = self.mesh_renderer(
            geom_utils.inverse_rt(mat=jTc, return_mat=True), intrinsics, jHand, **render_kwargs
        )

        # volumetric rendering
        # rays in canonical object frame
        rays_o, rays_d, select_inds = rend_util.get_rays(
            jTc, intrinsics, H, W, N_rays=-1)
        rgb_obj, depth_v, extras = render_fn(rays_o, rays_d, detailed_output=True, **render_kwargs)

        iObj = {'rgb': rgb_obj, 'depth': depth_v, }
        if use_surface_render:
            iObj['mask'] = extras['mask_surface']
        else:
            iObj['mask'] = extras['mask_volume']

        iHoi = self.blend(iHand, iObj, select_inds, zfar, znear, method='hard')
        
        # matting
        pred_hand_select = (iHand['depth'] < iObj['depth']).float().unsqueeze(-1)
        
        if not self.training:
            rgb = iHoi['rgb'].reshape(1, H, W, 3).permute([0, 3, 1, 2])
            pred_hand_select = pred_hand_select.reshape(1, 1, H, W)
        rtn = {}
        rtn['image'] = rgb.cpu()
        rtn['hand'] = iHand['image'].cpu()
        rtn['obj'] = iObj['rgb'].reshape(1, H, W, 3).permute([0, 3, 1, 2]).cpu()
        rtn['hand_front'] = pred_hand_select.cpu()
        rtn['obj_front'] = 1-pred_hand_select.cpu()
        rtn['label'] = iHoi['label'].reshape(1, H, W, 3).permute([0, 3, 1, 2])
        return rtn
    
    def sample_jTc_like(self, jTc):
        return mesh_utils.sample_camera_extr_like(wTc=jTc, t_std=self.args.novel_view.t_std)

    def sample_jTc(self, indices, model_input, ground_truth):
        jTc, jTc_n, jTh, jTh_n = self.get_jTc(indices, model_input, ground_truth)
        # jTc = mesh_utils.sample_camera_extr_like(wTc=jTc)  
        jTc = self.sample_jTc_like(jTc)
        # TODO: should do this but it breaks forward()
        # jTc_n = None
        return jTc, jTc_n, jTh, jTh_n 

    def get_jTc(self, indices, model_input, ground_truth):
        device = self.device
        # palmTcam
        wTc = self.posenet(indices, model_input, ground_truth, )
        wTc_n = self.posenet(model_input['inds_n'].to(device), model_input, ground_truth, )

        hTw = geom_utils.inverse_rt(mat=model_input['wTh'].to(device),return_mat=True)
        wTh = model_input['wTh'].to(device)
        hTw_n = geom_utils.inverse_rt(mat=model_input['wTh_n'].to(device),return_mat=True)
        
        # TODO: change here!
        oTh = self.model.oTh(indices.to(device), model_input, ground_truth)
        oTh_n = self.model.oTh(model_input['inds_n'].to(device), model_input, ground_truth)

        onTo = model_input['onTo'].to(device)
        onTo_n = model_input['onTo_n'].to(device)

        oTc = oTh @ hTw @ wTc
        oTc_n = oTh_n @ hTw_n @ wTc_n         

        # NOTE: the mesh / vol rendering needs to be in the same coord system (joint), in order to compare their depth!!
        if self.joint_frame == 'object_norm':
            jTh = onTo @ oTh
            jTh_n = onTo_n @ oTh_n 
            jTc = onTo @ oTc # wTc
            jTc_n = onTo @ oTc_n # wTc_n
        elif self.joint_frame == 'hand_norm':
            jTh =  wTh
            jTh_n = None
            jTc = wTc
            jTc_n = wTc_n

        return jTc, jTc_n, jTh, jTh_n

    def blend(self, iHand, iObj, select_inds, znear, zfar, 
            method='hard', sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0), **kwargs):
        N = len(iHand['image'])
        # change and select inds
        # iHand['rgb'] = iHand['image'].view(N, 3, -1).transpose(-1, -2)
        iHand['rgb']    = rearrange(iHand['image'], 'n c h w -> n (h w) c') 
        iHand['normal'] = rearrange(iHand['normal'], 'n c h w -> n (h w) c')
        iHand['depth'] = iHand['depth'].view(N, -1)
        iHand['mask'] = iHand['mask'].view(N, -1)

        iHand['rgb'] = torch.gather(iHand['rgb'], 1, torch.stack(3*[select_inds],-1))
        iHand['normal'] = torch.gather(iHand['normal'], 1, torch.stack(3*[select_inds],-1))
        iHand['depth'] = torch.gather(iHand['depth'], 1, select_inds).float()
        iHand['mask'] = torch.gather(iHand['mask'], 1, select_inds).float()

        blend_params = BlendParams(sigma, gamma, background_color)
        blend_label = BlendParams(sigma, gamma, (0., 0., 0.))
        iHoi = {}
        if method=='soft':
            # label: 
            hand_label = F.one_hot(torch.zeros_like(iHand['mask']).long(), num_classes=3).float()
            obj_label = F.one_hot(torch.ones_like(iObj['mask']).long(), num_classes=3).float()
            if self.args.training.label_prob == 2:
                hand_label *= iHand['mask'].unsqueeze(-1)
                obj_label *= iObj['mask'].unsqueeze(-1)
            iHoi['label'] = softmax_rgb_blend(
                (hand_label, obj_label),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_label, znear, zfar)[..., 0:3]

            rgba = softmax_rgb_blend(
                (iHand['rgb'], iObj['rgb']),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_params, znear, zfar)
            iHoi['rgb'], iHoi['mask'] = rgba.split([3, 1], -1)
            iHoi['mask'] = iHoi['mask'].squeeze(-1)
        elif method == 'vol':
            hand_label = F.one_hot(torch.zeros_like(iHand['mask']).long(), num_classes=3).float()
            obj_label = F.one_hot(torch.ones_like(iObj['mask']).long(), num_classes=3).float()
            iHoi['label'] = volumetric_rgb_blend(
                (hand_label, obj_label),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_label, znear, zfar)[..., 0:3]

            rgba = volumetric_rgb_blend(
                (iHand['rgb'], iObj['rgb']),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_params, znear, zfar)
            iHoi['rgb'], iHoi['mask'] = rgba.split([3, 1], -1)
            iHoi['mask'] = iHoi['mask'].squeeze(-1)
        elif method=='hard':
            rgba = hard_rgb_blend(
                (iHand['rgb'], iObj['rgb']),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_params)
            iHoi['rgb'], iHoi['mask'] = rgba.split([3, 1], -1)
            iHoi['mask'] = iHoi['mask'].squeeze(-1)

            hand_label = torch.zeros_like(iHand['mask']).long()
            obj_label = torch.ones_like(iHand['mask']).long()
            iHoi['label'] = hard_rgb_blend(
                (F.one_hot(hand_label, num_classes=3), F.one_hot(obj_label, num_classes=3)),
                (iHand['depth'], iObj['depth']),
                (iHand['mask'], iObj['mask']),
                blend_label)[..., 0:3]
        else:
            raise NotImplementedError('blend method %s' % method)        
        # placholder:
        # TODO: need to correct this :p 
        iHoi['flow'] = torch.zeros([N, iHoi['mask'].size(1), 2]).to(iHoi['mask'].device)
        return iHoi

    def get_reproj_loss(self, losses, extras, ground_truth, model_input, render_kwargs_train):
        """losses that compare with input -- cannot be applied to novel-view 

        :param iHoi: _description_
        :param ground_truth: _description_
        :param model_input: _description_
        :param select_inds: _description_
        :param losses: _description_
        :param render_kwargs_train: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        args = self.args
        device = self.device
        iHoi = extras['iHoi']
        select_inds = extras['select_inds']
        label_target = extras['label_target']
        target_rgb = extras['target_rgb']
        target_mask = extras['target_mask']
        select_inds = extras['select_inds']

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        elif args.training.fg == 1:
            mask_ignore = target_mask # masks for RGB / flow 
        else:
            mask_ignore = None

        if args.training.w_rgb > 0:
            losses['loss_img'] = F.l1_loss(iHoi['rgb'], target_rgb, reduction='none')
            # TODO: unify mask_ignore
            if mask_ignore is not None:
                losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
            else:
                losses['loss_img'] = losses['loss_img'].mean()

        if args.training.occ_mask == 'label':
            # deprecate the rest in Jan14
            # TODO Q1: GT(1, 1). --> pred??; Q2: normalize?? 
            losses['loss_mask'] = args.training.w_mask * F.l1_loss(iHoi['label'], label_target)
            # 
        else:
            raise NotImplementedError(args.training.occ_mask)

        max_H = max(render_kwargs_train['H'], render_kwargs_train['W'])
        if args.training.w_flow > 0:    
            # [B, N_rays, 2]
            target_flow_fw = torch.gather(model_input['flow_fw'].to(device), 1, torch.stack(2*[select_inds], -1))
            losses['loss_fl_fw'] = (2/max_H)**2 * args.training.w_flow * F.mse_loss(iHoi['flow'], target_flow_fw.detach(), reduction='none')
            if mask_ignore is not None:
                losses['loss_fl_fw'] = (losses['loss_fl_fw'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10) 
            else:
                losses['loss_fl_fw'] = losses['loss_fl_fw'].mean()

        if args.training.w_contour > 0:
            iHand = extras['iHand']
            gt_cont = model_input['hand_contour'].to(device)
            x_nn = knn_points(gt_cont, iHand['xy'][..., :2], K=1)
            cham_x = x_nn.dists.mean()
            losses['loss_contour'] = args.training.w_contour * cham_x
        return losses

    def get_reg_loss(self, losses, extras):
        """can be applied to original and novel view"""
        args = self.args
        # if  args.training.w_eikonal > 0:
        # always compute this even when the weight is zero --> populate loss to a correct shape
        nablas_norm = extras['implicit_nablas_norm']
        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

        # contour of hand masks
        # gt to pred --> just wish gt to be covered
        # contact loss
        if args.training.w_contact > 0:
            jHand = extras['hand']
            jVerts = jHand.verts_padded()            
            losses['contact'] = 0
            for i in range(6):
                 inds = getattr(self.hand_wrapper, 'contact_index_%d' % i)  # (V1, )
                 jContact = self.model.implicit_surface.forward(jVerts[:, inds])  # (N, V1, )
                 loss = torch.sum(torch.min(jContact, dim=1)[0].clamp(min=0))
                 losses['contact_%d' % i] = args.training.w_contact * loss
                 losses['contact'] += losses['contact_%d' % i]
    
    def get_label(self, iHoi, iHand, iObj):
        """ready to be consumed by sds
        read label, rescale, 

        :param iHoi: _description_
        :param iHand: _description_
        :param iObj: _description_
        """
        nv = self.args.novel_view
        if nv.amodal == 'occlusion':
            img = self.lin2img(iHoi['label'])
        elif nv.amodal == 'amodal':
            img = self.lin2img(torch.stack([
                iHand['mask'], iObj['mask'], torch.zeros_like(iHand['mask']),
            ], -1))
        img = img * 2 - 1 # rescale from 0/1 to -1/1
        return img

    def get_normal(self, iHoi, iHand, iObj, jTc, jTh):
        """ready to be consumed by sds (N, C, H, W)
        read normal, handle background
        !!! transform cordinate to either hand or camera !!! 
        normalize to 1 

        hand: 
        object: comes naturally in world coordinate? 

        :param iHoi: _description_
        :param iHand: _description_
        :param iObj: _description_
        """
        hTj = geom_utils.inverse_rt(mat=jTh, return_mat=True)
        cTj = geom_utils.inverse_rt(mat=jTc, return_mat=True)

        nv = self.args.novel_view
        if nv.amodal == 'occlusion':
            raise NotImplementedError(nv.amodal)
        elif nv.amodal == 'amodal':
            torch.autograd.set_detect_anomaly(True)
            cHand_normal = iHand['normal'] * iHand['mask'][..., None]

            jObj_normal = iObj['normal'] * iObj['mask'][..., None]  # (N, R, 3?)
            cTj_trans = Transform3d(matrix=cTj.transpose(-1, -2), device=cTj.device)
            # this is because my weird def of pytorch3d mesh rendering of normal in mesh_utils.render
            cObj_normal = cTj_trans.transform_normals(jObj_normal)    # need to renorm! coz of scale happens in cTj... 
            cObj_normal[..., 1:3] *= -1 # xyz points to left, up, outward
            cHoi_norm = torch.norm(cObj_normal, dim=-1, keepdim=True) + 1e-5
            cObj_normal = cObj_normal / cHoi_norm

            # cObj_normal = F.normalize(cObj_normal, -1)

            cHoi_normal = torch.cat([cHand_normal, cObj_normal], -1)  # (N, R, 32)

            if nv.frame == 'hand':
                hTc =hTj @  jTc 
                hTc_trans = Transform3d(matrix=hTc.transpose(-1, -2), device=hTc.device)
                # TODO: maybe a bug
                cHoi_lin = rearrange(cHoi_normal, 'n r (c k) -> n (r k) c', c=3)  
                hHoi_lin = hTc_trans.transform_normals(cHoi_lin)
                hHoi_norm = torch.norm(hHoi_lin, dim=-1, keepdim=True) + 1e-10
                hHoi_lin /= hHoi_norm
                # there is a scale factor so norm will change~~
                # H = cHand_normal.shape[-1]
                hHoi_normal = rearrange(hHoi_lin, 'n (r k) 3 -> n r (3 k)',)
                img = self.lin2img(hHoi_normal)

            elif nv.frame == 'camera':
                img = self.lin2img(cHoi_normal)  # N, 6, H, W

        return img

    def get_depth(self, iHoi, iHand, iObj, jTc, jTh, zfar=-10):
        """eady to be consumed by sds (N, C, H, W)
        read depth, 
        handle background??
        get relative depth to center of hand
        !!! transform cordinate to either hand or camera !!! 
        !!! rescale the metric to 1=100mm/0.1m

        :param iHoi: _description_
        :param iHand: _description_
        :param iObj: _description_
        :param cTj: _description_
        :param hTj: _description_
        """
        nv = self.args.novel_view
        N = len(jTc)
        if nv.amodal == 'occlusion':
            raise NotImplementedError(nv.amodal)
        elif nv.amodal == 'amodal':
            _, _, jTc_scale = geom_utils.homo_to_rt(jTc)  # (N, 3)            
            jTc_scale = jTc_scale.mean(-1).reshape(N, 1) / 10  # *10: m -> dm

            hand_mask = (iHand['mask'] > 0.5).float()
            obj_mask = (iObj['mask'] > 0.5).float()
            
            jHand_depth_camera = iHand['depth'] * hand_mask
            jObj_depth_camera = iObj['depth'] * obj_mask
            
            cHand_depth_camera = jHand_depth_camera / jTc_scale
            cObj_depth_camera = jObj_depth_camera / jTc_scale

            mean_hand_depth = cHand_depth_camera.sum(1, keepdim=True) / hand_mask.sum(1, keepdim=True)            
            
            cHand_reldepth = hand_mask * (cHand_depth_camera - mean_hand_depth) + (1-hand_mask) * zfar
            cObj_reldepth = obj_mask * (cObj_depth_camera-mean_hand_depth) + (1 - obj_mask) * zfar

            cHoi_reldepth = torch.stack([cHand_reldepth, cObj_reldepth], -1)
            cHoi_reldepth = self.lin2img(cHoi_reldepth)  # (N, C, H, W? )
            
            img = cHoi_reldepth

        return img

    def get_diffusion_image(self, extras):
        args = self.args
        nv = args.novel_view
        iHoi = extras['iHoi']
        iHand = extras['iHand']
        iObj = extras['iObj']
        rtn = {}
        if nv.mode == 'semantics':
            img = self.get_label(iHoi, iHand, iObj)
        elif nv.mode == 'geom':
            zfar = self.sd_loss.model.cfg.zfar
            mask = self.get_label(iHoi, iHand, iObj)
            normal = self.get_normal(iHoi, iHand, iObj, extras['jTc'], extras['jTh'])
            depth = self.get_depth(iHoi, iHand, iObj, extras['jTc'], extras['jTh'], zfar)

            if not self.sd_loss.cond:
                img = torch.cat([mask, normal, depth], 1)
                rtn['image'] = img
            else:
                hand_mask, obj_mask = mask[:, 0:1], mask[:, 1:2]
                hand_normal, obj_normal = normal.split([3, 3], 1)
                hand_depth, obj_depth = depth.split([1, 1], 1)

                rtn['image'] = torch.cat([obj_mask, obj_normal, obj_depth], 1)        
                rtn['cond_image'] = torch.cat([hand_mask, hand_normal, hand_depth], 1)
        else: 
            raise NotImplementedError(nv.mode)
        return rtn


    def get_fullframe_reg_loss(self, losses, extras):
        """can only be applied to full_frame rendering, esp for novel view"""
        # Apply Distilled Score from dream diffusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
        args = self.args
        if args.training.w_diffuse > 0:
            img = self.get_diffusion_image(extras)
            self.sd_loss.apply_sd(**img, 
                weight=args.training.w_diffuse, **args.novel_view.loss)
            extras['diffusion_inp'] = img
        return losses

    def get_temporal_loss(self, losses, extras):
        """only makes sense when consective frame make sense~"""
        args = self.args
        # temporal smoothness loss
        # jJoints, hJoints? 
        if args.training.w_t_hand > 0:
            jJonits_diff = ((extras['jJoints'] - extras['jJoints_n'])**2).mean()
            cJonits_diff = ((extras['cJoints'] - extras['cJoints_n'])**2).mean()
            losses['loss_dt_joint'] = 0.5 * args.training.w_t_hand * (jJonits_diff + cJonits_diff)
        return losses

    def forward(self, 
             args,
             indices,
             model_input,
             ground_truth,
             render_kwargs_train: dict,
             it: int,
             calc_loss=True,
             ):
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
        cam_norm = jTc[..., 0:4, 3]
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
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        intrinsics = self.focalnet(indices, model_input, ground_truth, H=H, W=W)
        intrinsics_n = self.focalnet(model_input['inds_n'].to(device), model_input, ground_truth, H=H,W=W)

        if full_frame_iter:
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2

        # 2. RENDER MY SCENE 
        # 2.1 RENDER OBJECT
        # volumetric rendering
        # rays in canonical object frame
        if self.training:
            if full_frame_iter:
                rays_o, rays_d, select_inds = rend_util.get_rays(
                    jTc, intrinsics, H, W, N_rays=-1)
            else:
                rays_o, rays_d, select_inds = rend_util.get_rays(
                    jTc, intrinsics, H, W, N_rays=args.data.N_rays)
        else:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                jTc, intrinsics, H, W, N_rays=-1)
        
        rgb_obj, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)
        # rgb: (N, R, 3), mask/depth: (N, R, ), flow: (N, R, 2?)
        iObj = {'rgb': rgb_obj, 'depth': depth_v, 'mask': extras['mask_volume'],}
        if jTc_n is not None:
            iObj['flow'] = flow_12 = volume_render_flow(depth_v, select_inds, intrinsics, intrinsics_n, jTc, jTc_n, **render_kwargs_train)
        if 'normals_volume' in extras:
            iObj['normal'] = extras['normals_volume']
        extras['iObj'] = iObj
        extras['select_inds'] = select_inds

        # 2.2 RENDER HAND
        # hand FK
        hA = self.model.hA_net(indices, model_input, None)
        hA_n = self.model.hA_net(model_input['inds_n'].to(device), model_input, None)
        hHand, hJoints = self.hand_wrapper(None, hA, texture=self.model.uv_text)
        jHand = mesh_utils.apply_transform(hHand, jTh)
        jJoints = mesh_utils.apply_transform(hJoints, jTh)
        cJoints = mesh_utils.apply_transform(jJoints, geom_utils.inverse_rt(mat=jTc, return_mat=True))
        extras['jTc'] = jTc
        extras['jTh'] = jTh
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

        if not calc_loss:
            # early return 
            return extras
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
        # extras['flow'] = iHoi['flow']
        extras['hA'] = hA
        extras['intrinsics'] = intrinsics

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
    
    def render_novel_view(self, indices, model_input, ground_truth, render_kwargs_test):
        H, W = render_kwargs_test['H'], render_kwargs_test['W']
        jHand, jTc, _, intrinsics = self.sample_jHand_camera(indices, model_input, ground_truth, H, W)
        image1 = self.render(jHand, jTc, intrinsics, render_kwargs_test)
        return image1

    def val(self, logger: Logger, ret, to_img_fn, it, render_kwargs_test, val_ind=None, val_in=None, val_gt=None):        
        mesh_utils.dump_meshes(osp.join(logger.log_dir, 'hand_meshes/%08d' % it), ret['hand'])
        logger.add_meshes('hand',  osp.join('hand_meshes/%08d_0.obj' % it), it)
        
        # render a novel view? 
        novel_img = self.render_novel_view(val_ind, val_in, val_gt, render_kwargs_test)
        logger.add_imgs(novel_img['image'], 'sample/hoi_rgb', it)
        logger.add_imgs(novel_img['label'].float(), 'sample/hoi_label', it)

        # get_diffusion image
        if self.sd_loss is not None:
            image = self.get_diffusion_image(ret)
            for kk, vv in image.items():
                out = self.sd_loss.model.decode_samples(vv)
                for k,v in out.items():
                    if 'normal' in k:
                        v = v / 2 + 0.5
                    if 'depth' in k:
                        color = image_utils.save_depth(v, None, znear=-1, zfar=1)
                        logger.add_imgs(TF.to_tensor(color)[None], f'diffuse/{k}_{kk}_color', it)
                        
                    logger.add_imgs(v.clip(-1, 1), f'diffuse/{k}_{kk}', it)

            # try free generation? 
            noise_step = self.sd_loss.max_step
            noisy = self.sd_loss.get_noisy_image(image['image'], noise_step-1)
            out = self.sd_loss.vis_multi_step(noisy, noise_step, 1, image.get('cond_image', None))
            out = self.sd_loss.model.decode_samples(out)
            for k,v in out.items():
                if 'normal' in k:
                    v = v / 2 + 0.5
                if 'depth' in k:
                    color = image_utils.save_depth(v, None, znear=-1, zfar=1)
                    logger.add_imgs(TF.to_tensor(color)[None], f'diffuse_sample/{k}_color', it)
                    
                logger.add_imgs(v.clip(-1, 1), f'diffuse_sample/{k}', it)

        color = image_utils.save_depth(to_img_fn(ret['iObj']['depth'].unsqueeze(-1)), None, znear=-1, zfar=1)
        logger.add_imgs(TF.to_tensor(color)[None], f'diffuse/obj_depth_nomask', it)
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
        
        depth_hand = plot_utils.pc_to_cubic_meshes(pc=depth_hand, eps=5e-2)
        depth_obj = plot_utils.pc_to_cubic_meshes(pc=depth_obj, eps=5e-2)

        mesh_utils.dump_meshes(
            osp.join(logger.log_dir, 'meshes/%08d_hand' % it), 
            depth_hand)
        mesh_utils.dump_meshes(osp.join(logger.log_dir, 'meshes/%08d_obj' % it), 
            depth_obj)

        depth_hoi = mesh_utils.join_scene_w_labels([depth_hand, depth_obj], 3)
        image_list = mesh_utils.render_geom_rot(depth_hoi, 'circle', cameras=cameras, view_centric=True)
        logger.add_gifs(image_list, 'hoi/hoi_depth_pointcloud', it)


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
        'calc_normal': True,
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    render_kwargs_test['calc_normal'] = True

    devices = kwargs.get('device', args.device_ids)
    trainer = Trainer(model, devices, batched=render_kwargs_train['batched'], args=args)
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer, volume_render_flow
