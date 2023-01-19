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

class Glide(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.template_size = [cfg.ndim, cfg.side_y, cfg.side_x]
    
    def init_model(self,):
        cfg =self.cfg.model
        glide_model, glide_diffusion, glide_options = glide_util.load_model(
            glide_path=cfg.resume_ckpt,
            use_fp16=self.cfg.use_fp16,
            disable_transformer=cfg.disable_transformer,
            freeze_transformer=cfg.freeze_transformer,
            freeze_diffusion=cfg.freeze_diffusion,
            activation_checkpointing=cfg.activation_checkpointing,
            model_type='base',        
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


class GeomGlide(Glide):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

    def decode_samples(self, tensor):
        masks, hand_normal, obj_normal,  hand_depth, obj_depth = \
            tensor.split([3, 3, 3, 1, 1], 1)
        return {
            'semantics': masks, 
            'hand_normal': hand_normal,
            'obj_normal': obj_normal,
            'hand_depth': hand_depth,
            'obj_depth': obj_depth,
        }

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        
        out = self.decode_samples(samples)
        for k, v in out.items():
            log[f"{pref}sample_{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        out = self.decode_samples(batch['image'])
        for k, v in out.items():
            log[f"{pref}{k}"] = wandb.Image(vutils.make_grid(v, value_range=[-1, 1]))
        return log