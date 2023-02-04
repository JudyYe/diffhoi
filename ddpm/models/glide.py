# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Created on Sat Sep 17 2022
# --------------------------------------------------------
import wandb
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_lightning.utilities.distributed import rank_zero_only

from .glide_base import BaseModule
from ..utils import glide_util
from jutils import image_utils, model_utils

class Glide(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.template_size = [cfg.ndim, cfg.side_y, cfg.side_x]
    
    def init_model(self,):
        cfg =self.cfg.model
        if self.cfg.mode.cond:
            model_type='base-inpaint'
        else:
            model_type='base'
        glide_model, glide_diffusion, glide_options = glide_util.load_model(
            glide_path=cfg.resume_ckpt,
            use_fp16=self.cfg.use_fp16,
            disable_transformer=cfg.disable_transformer,
            freeze_transformer=cfg.freeze_transformer,
            freeze_diffusion=cfg.freeze_diffusion,
            activation_checkpointing=cfg.activation_checkpointing,
            model_type=model_type,
            in_channels=self.cfg.ndim,
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
        return log

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        out = self.decode_samples(batch['image'])
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log
    

class CondGeomGlide(Glide):
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

        print('x_t', x_t.shape, 'cond_image', batch['cond_image'].shape)
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
        
        out = self.decode_samples(batch['cond_image'])
        for k, v in out.items():
            if 'depth' in k:
                log[f"{pref}{k}_color_cond"] = wandb.Image(image_utils.save_depth(v, None, znear=-1, zfar=1))
            log[f"{pref}{k}_cond"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log