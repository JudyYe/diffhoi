# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Created on Sat Sep 17 2022
# --------------------------------------------------------
import os.path as osp
import wandb
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch3d.renderer.cameras import PerspectiveCameras
from .glide_base import BaseModule
from ..utils import glide_util
from jutils import image_utils, model_utils, mesh_utils, geom_utils, plot_utils

class Glide(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.template_size = [cfg.ndim, cfg.side_y, cfg.side_x]

    def decode_samples(self, tensor):
        return {
            'semantics': tensor, 
        }
        
    def init_model(self,):
        cfg =self.cfg.model
        model_type='base-inpaint' if self.cfg.ndim_cond > 0 else 'base'
        glide_model, glide_diffusion, glide_options = glide_util.load_model(
            glide_path=cfg.resume_ckpt,
            use_fp16=self.cfg.use_fp16,
            disable_transformer=cfg.disable_transformer,
            freeze_transformer=cfg.freeze_transformer,
            freeze_diffusion=cfg.freeze_diffusion,
            activation_checkpointing=cfg.activation_checkpointing,
            model_type=model_type,
            in_channels=self.cfg.ndim,
            cond_channels=self.cfg.ndim_cond
        )
        self.glide_model = glide_model
        self.diffusion = glide_diffusion
        self.glide_options = glide_options

        if self.cfg.resume_ckpt is not None and self.cfg.resume_ckpt.endswith('.ckpt'):
            sd = torch.load(self.cfg.resume_ckpt, map_location="cpu")
            if 'state_dict' in sd:
                sd = sd['state_dict']
            missing, unexpected = self.load_state_dict(sd, strict=False)
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")

        return glide_model, glide_diffusion, glide_options

    def step(self, batch, batch_idx):
        device = self.device
        glide_model = self.glide_model
        glide_diffusion = self.diffusion
        tokens, masks, reals = batch['token'], batch['token_mask'], batch['image']

        timesteps = torch.randint(
            0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
        )
        batch_size = len(masks)
        noise = torch.randn([batch_size,] + self.template_size, device=device)
        x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise,
            ).to(device)
        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            mask=masks.to(device),
            tokens=tokens.to(device),
        )
        epsilon = model_output[:, :model_output.shape[1]//2]
        loss = F.mse_loss(epsilon, noise.to(device).detach())        
        return loss, {'loss': loss}

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        
        out = self.decode_samples(samples)
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}sample_{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log


class GeomGlide(Glide):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

    # @classmethod
    def decode_samples(self, tensor):
        mode = self.cfg.mode
        out = {}
        cur = 0
        if mode.mask:
            out['semantics'] = tensor[:, cur:cur+3]
            cur += 3
        if mode.normal:
            out['hand_normal'] = tensor[:, cur:cur+3]
            out['obj_normal'] = tensor[:, cur+3:cur+6]
            cur += 6
        if mode.depth:
            out['hand_depth'] = tensor[:, cur:cur+1]
            out['obj_depth'] = tensor[:, cur+1:cur+2]
            cur += 2
        return out
    
    def distribute_weight(self, grad, w_mask, w_normal, w_depth, *args, **kwargs):
        grad[:, 0:3] *= w_mask
        grad[:, 3:3+3*2] *= w_normal
        grad[:, 3+3*2:3+3*2+1*2] *= w_depth
        return grad

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        

        out = self.decode_samples(samples)
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}sample_{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
    
        log = self.vis_depth_as_pc(out['hand_depth'], out['obj_depth'], f'{pref}sample', log, step, 
                                   out['semantics'][:, 1], out['semantics'][:, 0])
        return log

    def vis_depth_as_pc(self, depth_hand, depth_obj,pref, log, step, mask_hand=None, mask_obj=None, f=10):
        device= self.device
        if step is None:
            step = self.global_step
        # move z to around 2focal 
        dist = 2*f
        depth_hand = depth_hand +  dist
        depth_obj = depth_obj + dist
        cameras = PerspectiveCameras(f, device=self.device)
        depth_hand_pc = mesh_utils.depth_to_pc(depth_hand.to(self.device), cameras=cameras)
        depth_obj_pc = mesh_utils.depth_to_pc(depth_obj.to(self.device), cameras=cameras)
        
        depth_hand_mesh = plot_utils.pc_to_cubic_meshes(pc=depth_hand_pc, eps=5e-2)
        depth_obj_mesh = plot_utils.pc_to_cubic_meshes(pc=depth_obj_pc, eps=5e-2)

        mesh_utils.dump_meshes(
            osp.join(self.log_dir, f'meshes/{step:08d}_{pref}_hand'), 
            depth_hand_mesh)
        mesh_utils.dump_meshes(
            osp.join(self.log_dir, f'meshes/{step:08d}_{pref}_obj'), 
            depth_obj_mesh)
        
        depth_hand_fg = mesh_utils.depth_to_pc(depth_hand.to(self.device), cameras=cameras, mask=mask_hand.to(device),)
        depth_obj_fg = mesh_utils.depth_to_pc(depth_obj.to(self.device), cameras=cameras, mask=mask_obj.to(device))
        depth_hand_fg = depth_hand_fg.points_list()[0]
        depth_obj_fg = depth_obj_fg.points_list()[0]
        color_hand_fg = torch.zeros_like(depth_hand_fg); color_hand_fg[..., 1] = 255
        color_obj_fg = torch.zeros_like(depth_obj_fg); color_obj_fg[..., 0] = 255
        hoi_point = torch.cat([
            torch.cat([depth_hand_fg, color_hand_fg], -1),
            torch.cat([depth_obj_fg, color_obj_fg], -1),
        ], -2)
        log[f"{pref}pc"] = wandb.Object3D(hoi_point.cpu().detach().numpy())

        depth_hoi = mesh_utils.join_scene_w_labels([depth_hand_mesh, depth_obj_mesh], 3)
        image_list = mesh_utils.render_geom_rot(depth_hoi, 'circle', cameras=cameras, view_centric=True)
        image_utils.save_gif(image_list, osp.join(self.log_dir, f'gifs/{step:08d}_{pref}_hoi'))
        log[f'{pref}gif_pc'] = wandb.Video(osp.join(self.log_dir, f'gifs/{step:08d}_{pref}_hoi') + '.gif')

        return log

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        out = self.decode_samples(batch['image'])
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        log = self.vis_depth_as_pc(out['hand_depth'], out['obj_depth'], f'{pref}gt', log, step,
                                   out['semantics'][:, 1], out['semantics'][:, 0])
        return log
    

class ObjGeomGlide(Glide):

    def step(self, batch, batch_idx):
        device = self.device
        glide_model = self.glide_model
        glide_diffusion = self.diffusion
        batch = model_utils.to_cuda(batch, device)

        tokens, masks, reals = batch['token'], batch['token_mask'], batch['image']

        timesteps = torch.randint(
            0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
        )
        batch_size = len(masks)
        noise = torch.randn([batch_size,] + self.template_size, device=device)
        x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise,
            ).to(device)

        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            mask=masks.to(device),
            tokens=tokens.to(device),
        )
        epsilon = model_output[:, :model_output.shape[1]//2]
        loss = F.mse_loss(epsilon, noise.to(device).detach())        
        return loss, {'loss': loss}

    # @classmethod
    def decode_samples(self, tensor):
        mode = self.cfg.mode
        out = {}
        cur = 0
        if mode.mask:
            out['semantics'] = tensor[:, cur:cur+1]
            cur += 1
        if mode.normal:
            out['obj_normal'] = tensor[:, cur:cur+3]
            cur += 3
        if mode.depth:
            out['obj_depth'] = tensor[:, cur:cur+1]
            cur += 1
        return out
    
    def distribute_weight(self, grad, w_mask, w_normal, w_depth, *args, **kwargs):
        grad[:, 0:1] *= w_mask
        grad[:, 1:1+3] *= w_normal
        grad[:, 1+3:1+3+1] *= w_depth
        return grad

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        
        out = self.decode_samples(samples)
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}sample_{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        out = self.decode_samples(batch['image'])
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log


class CondGeomGlide(GeomGlide):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

    def step(self, batch, batch_idx):
        device = self.device
        glide_model = self.glide_model
        glide_diffusion = self.diffusion
        batch = model_utils.to_cuda(batch, device)

        tokens, masks, reals = batch['token'], batch['token_mask'], batch['image']

        timesteps = torch.randint(
            0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
        )
        batch_size = len(masks)
        noise = torch.randn([batch_size,] + self.template_size, device=device)
        x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise,
            ).to(device)

        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            mask=masks.to(device),
            tokens=tokens.to(device),
            cond_image=batch['cond_image'],
        )
        epsilon = model_output[:, :model_output.shape[1]//2]
        loss = F.mse_loss(epsilon, noise.to(device).detach())        
        return loss, {'loss': loss}

    # @classmethod
    def decode_samples(self, tensor):
        mode = self.cfg.mode
        out = {}
        cur = 0
        if mode.mask:
            out['semantics'] = tensor[:, cur:cur+1]
            cur += 1
        if mode.normal:
            out['obj_normal'] = tensor[:, cur:cur+3]
            cur += 3
        if mode.depth:
            out['obj_depth'] = tensor[:, cur:cur+1]
            cur += 1
        if mode.uv:
            # only do hand
            if cur < tensor.shape[1]:
                out['obj_uv'] = torch.cat(
                    [tensor[:, cur:cur+2], torch.zeros_like(tensor[:, cur:cur+1])], 
                1)
                cur += 2
        return out
    
    def distribute_weight(self, grad, w_mask, w_normal, w_depth, *args, **kwargs):
        grad[:, 0:1] *= w_mask
        grad[:, 1:1+3] *= w_normal
        grad[:, 1+3:1+3+1] *= w_depth
        return grad

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        
        out = self.decode_samples(samples)
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}sample_{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        out_cond = self.decode_samples(batch['cond_image'])

        log = self.vis_depth_as_pc(out_cond['obj_depth'], out['obj_depth'], f'{pref}sample', log, step,
                                   out_cond['semantics'], out['semantics'])
        return log

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        out = self.decode_samples(batch['image'])
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        
        out_cond = self.decode_samples(batch['cond_image'])
        for k, v in out_cond.items():
            if 'depth' in k:
                log[f"{pref}{k}_color_cond"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}{k}_cond"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        log = self.vis_depth_as_pc(out_cond['obj_depth'], out['obj_depth'], f'{pref}gt', log, step,
                                   out_cond['semantics'], out['semantics'])
        return log