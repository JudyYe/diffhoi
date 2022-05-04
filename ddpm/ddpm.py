import logging
import os
import shutil
import numpy as np
import math
import copy
import torch
import torch.distributed as dist
from torch import  nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from torch.cuda.amp import autocast, GradScaler

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from jutils import image_utils, geom_utils, mesh_utils, model_utils
from utils import io_util
from utils.checkpoints import CheckpointIO
from utils.dist_util import get_world_size, is_master
from utils.hand_utils import ManopthWrapper, get_nTh

from utils.logger import Logger
# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Upsample3D(dim):
    return nn.ConvTranspose3d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

def Downsample3D(dim):
    return nn.Conv3d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)


class ConvNextBlock3D(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv3d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv3d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1 1')  # 3D operation

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (z x y) -> b (h c) z x y', h = self.heads, z = d, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)  # b, h*c, x, y
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), qkv)  # sequeeze spatial dimension, split k-head dimension
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (z x y) d -> b (h d) z x y', z=d, x=h, y=w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock3D(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock3D(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample3D(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock3D(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ConvNextBlock3D(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock3D(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock3D(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample3D(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock3D(dim, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        starttime = 1000,
        loss_type = 'l1'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.start_time = starttime
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        # output noise, not mean. 
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c, clip_denoised: bool):
        # ??? conditionall??? 
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, c))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, c, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=c, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, img=None, context=None, t=None, q_sample=True):
        device = self.betas.device
        t = default(t, self.num_timesteps - 1)
        b = shape[0]
        if img is None:            
            img = torch.randn(shape, device=device)
        else:
            # add nosie on img
            b, *_, device = *img.shape, img.device
            
            t_batched = torch.stack([torch.tensor(t, device=device)] * b)
            if q_sample:
                img = self.q_sample(img, t=t_batched)

            assert b == len(img)

        for i in tqdm(reversed(range(0, t + 1)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), context)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16, **kwargs):
        image_size = self.image_size
        channels = self.channels

        context = self.get_context(**kwargs)

        return self.p_sample_loop((batch_size, channels, image_size, image_size, image_size),
            img=kwargs.get('img', None), context=context, t=kwargs.get('t', None), 
            q_sample=kwargs.get('q_sample', True))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # for train
    def q_sample(self, x_start, t, noise=None):
        # perturbe the Gaussian noise
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, context=None):
        b, c, d, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, context=context)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def get_context(self, **kwargs):
        hA = kwargs.get('hA', None)
        b = hA.shape[0]
        context = geom_utils.axis_angle_t_to_matrix(hA.reshape(b, 15, 3), homo=False).reshape(b, 1, 15 * 9)
        return context

    def forward(self, x, *args, **kwargs):
        """
        x: SDF in shape of (N, 1, D, H, W)
        """
        b, c, d, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert d ==img_size and h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.start_time, (b,), device=device).long()
        context = self.get_context(**kwargs)
        return self.p_losses(x, t, context=context)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        valset,
        *,
        device_ids=[0,],
        ema_decay = 0.995,
        train_batch_size = 32,
        test_batch_size=8,
        start_time=1000,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        logger: Logger=None,
        args=dict(),
    ):
        super().__init__()
        self.device = device_ids[0]
        self.args = args
        self.start_time = start_time
        self.hand_wrapper = ManopthWrapper().to(self.device)
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.test_bs = test_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.val_ds = valset
        if args.ddp:
            train_sampler = DistributedSampler(dataset)
            self.dl = cycle(data.DataLoader(self.ds, sampler=train_sampler, 
                batch_size = train_batch_size, pin_memory=True, num_workers=args.environment.workers))
            val_sampler = DistributedSampler(valset)
            self.val_dl = cycle(data.DataLoader(self.val_ds,sampler=val_sampler, 
                batch_size = test_batch_size,  pin_memory=True, num_workers=args.environment.workers))
        else:
            self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, 
                shuffle=True, pin_memory=True, num_workers=args.environment.workers))
            self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size = test_batch_size, 
                shuffle=False, pin_memory=True, num_workers=args.environment.workers))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()
        self.logger = logger

        self.checkpoint_io = CheckpointIO(checkpoint_dir=results_folder, allow_mkdir=is_master())
        if get_world_size() > 1:
            dist.barrier()
            
        self.checkpoint_io.register_modules(
            model=self.model,
            ema=self.ema_model,
            scaler=self.scaler,
            opt=self.opt,
        )
        if is_master():
            io_util.save_config(args, os.path.join(args.exp_dir, 'config.yaml'))

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        self.checkpoint_io.save(filename='model-{:08d}.pt'.format(milestone), step=self.step + get_world_size())
        os.system('rm %s' % str(self.results_folder / 'latest.pt'))
        cmd = 'ln -s %s %s' % (str(self.results_folder / 'model-{:08d}.pt'.format(milestone)), str(self.results_folder / 'latest.pt'))
        os.system(cmd)

    def load(self, milestone=None, **kwargs):
        load_dict = self.checkpoint_io.load_file(milestone, **kwargs)
        self.step = load_dict.get('step', 0)
        
    def train(self):
        special_ckpt = self.args.special_ckpt 
        device = self.device
        valdata = next(self.val_dl)
        with tqdm(range(self.train_num_steps), disable=not is_master()) as pbar:
            if is_master():
                pbar.update(self.step)
            while self.step < self.train_num_steps:
                for i in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    origin_sdf = data['nSdf']
                    with autocast(enabled = self.amp):
                        loss = self.model(data['nSdf'].to(device), hA=data['hA'].to(device))
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    self.logger.add('loss', 'total', loss, self.step)

                if is_master():
                    pbar.set_postfix(loss_total=loss.item())

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.args.n_eval_freq == 0 or self.step in special_ckpt:
                    origin_sdf = valdata['nSdf'].to(device)
                    nTh = get_nTh(hA=valdata['hA'].to(device), hand_wrapper=self.hand_wrapper)
                    nHand, _ = self.hand_wrapper(nTh, valdata['hA'].to(device))

                    self.vis_sdf(origin_sdf, 'initial', nHand)
                    noised_inp = self.ema_model.q_sample(origin_sdf, 
                        t=torch.stack([torch.tensor(self.start_time - 1, device=device)] * self.test_bs))
                    self.vis_sdf(noised_inp, 'initial_start', nHand)
                    noised_inp = self.ema_model.q_sample(origin_sdf, 
                        t=torch.stack([torch.tensor(self.start_time // 10, device=device)] * self.test_bs))
                    self.vis_sdf(noised_inp, 'initial_start10perc', nHand)

                    all_points_list = self.ema_model.sample(
                        self.test_bs, img=valdata['nSdf'].to(device), hA=valdata['hA'].to(device), 
                        t=self.start_time - 1)
                    self.vis_sdf(all_points_list, 'denoise_start', nHand)

                    all_points_list = self.ema_model.sample(
                        self.test_bs, img=valdata['nSdf'].to(device), hA=valdata['hA'].to(device), 
                        t=self.start_time // 10 - 1)
                    self.vis_sdf(all_points_list, 'denoise_start10perc', nHand)


                    all_points_list = self.ema_model.sample(
                        self.test_bs, hA=valdata['hA'].to(device), t=self.start_time - 1)
                    self.vis_sdf(all_points_list, 'geenrate', nHand)

                if is_master():
                    if self.step != 0 and self.step % self.save_and_sample_every == 0 or self.step in special_ckpt:
                    # if self.step % self.save_and_sample_every == 0:
                        milestone = self.step 
                        self.save(milestone)
                self.step += get_world_size()
                if is_master():
                    pbar.update(get_world_size())

            print('training completed')

    def vis_sdf(self, sdf, name, nHand=None):
        meshes = mesh_utils.batch_grid_to_meshes(sdf.squeeze(1), len(sdf))
        if nHand is not None:
            meshes = mesh_utils.join_scene([nHand, meshes])
        image_list = mesh_utils.render_geom_rot(meshes, scale_geom=True)
        self.logger.add_gifs(image_list, 'surface/%s' % name, self.step)
        image_list = self.view_vol_as_gifs(sdf)
        self.logger.add_gifs(image_list, 'slice/%s' % name, self.step)


    def view_vol_as_gifs(self, sdfs):
        """z
        sdfs: (N, 1, D, H, W)
        """
        N = len(sdfs)
        D = sdfs[0].shape[1]
        image_list = []
        for d in range(D):
            image_list.append(sdfs[:, :, d])
        return image_list

