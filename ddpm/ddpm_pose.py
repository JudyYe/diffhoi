import os

import os.path as osp
from omegaconf import OmegaConf
from ddpm.pointnet import PointNet2ClassificationSSG
from utils.dist_util import get_local_rank, get_rank, get_world_size, init_env, is_master

import torch
from torch import  nn
import torch.nn.functional as F


from .tutorial import default
from tqdm import tqdm
from jutils import image_utils, geom_utils, mesh_utils, model_utils
from utils.dist_util import get_world_size, is_master
from utils.logger import Logger
# helpers functions

from .ddpm import Trainer as BaseTrainer


class GaussianPoseDiffusion(nn.Module):
    def __init__(self, denoise_fn, *, image_size, loss_type='l1', xyz_std=0, theta_std=0):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.loss_type = loss_type
        self.xyz_std = xyz_std
        self.theta_std = theta_std
    
    def add_noise(self, x, noise, rep='axisang'):
        hand, obj = torch.chunk(x, 2, 1)
        xyz = obj[..., :3]
        if rep == 'axisang':
            trans = geom_utils.axis_angle_t_to_matrix(noise[..., 3:6], noise[..., 0:3])
        elif rep == 'se3':
            trans = geom_utils.se3_to_matrix(noise, no_scale=True)
        xyz = mesh_utils.apply_transform(xyz, trans)
        obj = torch.cat([xyz, obj[..., 3:]], -1)

        x_recon = torch.cat([hand, obj], 1)
        return x_recon

    @torch.no_grad()
    def p_sample(self, x, ):
        # x_t --> x_{t-1}???
        minus_noise = self.denoise_fn(x)  # se3        

        x_recon = self.add_noise(x, minus_noise, rep='se3')
        return x_recon

    @torch.no_grad()
    def p_sample_loop(self, x=None, t=100, condition_kwargs=None, q_sample=True, cond_scale=1):
        img = x
        device = self.betas.device

        # add nosie on img
        b, *_, device = *img.shape, img.device
        
        if q_sample:
            img = self.q_sample(img, )

        assert b == len(img)

        for i in tqdm(reversed(range(0, t + 1)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img)
        return img

    def q_sample(self, x ,noise=None):
        """
        :param x_start: (N, P*2, 3+D=3)
        :param t

        :return noised_x: (N, P*2, 3+D), when sampled from random se3
        """
        # random se3
        x_start = x
        device = x_start.device
        noise = default(noise, geom_utils.randn_4d(len(x_start), device))
        scale = torch.FloatTensor([[self.xyz_std, self.theta_std]]).to(device)  # (1, 2)
        scaled_noise = geom_utils.scalar_times_4d(scale, noise)

        noised_x = self.add_noise(x_start, scaled_noise)
        return noised_x

    def forward(self, x, noise = None, condition_kwargs={}, **kwargs):
        x_start = x
        x_noisy = self.q_sample(x=x_start, noise=noise)  # a point cloud, add noise on top of x_start
        # TODO: vis x_noisy
        minus_noise = self.denoise_fn(x_noisy)  # se3

        # TODO: t? t-1??)
        x_recon = self.add_noise(x_noisy, minus_noise, rep='se3')

        if self.loss_type == 'l1':
            loss = (x_start - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, x_recon)
        else:
            raise NotImplementedError()

        return loss


class Trainer(BaseTrainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        valset,
        args=dict(),
        **kwargs,
    ):
        super().__init__(
            diffusion_model,
            dataset,
            valset,
            args=args,
            **kwargs
        )

    def _change_hand_fix_shape(self, ):
        onedata = next(self.val_dl)
        bs = onedata['nObj_pc'].shape[0]
        first_shape = onedata['nObj_pc'][-2:-1].expand(*onedata['nObj_pc'].shape)
        onedata['nObj_pc'] = first_shape
        if hasattr(self.val_ds, 'special_hA'):
            b = len(self.val_ds.special_hA)
            onedata['hA'][:min(b, bs)] = self.val_ds.special_hA[:min(b, bs)]
        return onedata
    
    def vis_step(self):
        onedata = self.onedata
        valdata = self.valdata
        device = self.device

        # vis initial 
        noised_val = self.ema_model.q_sample(**self.set_input(valdata))
        self.vis_pc(noised_val, 'noised/initial')
        noised_one = self.ema_model.q_sample(**self.set_input(onedata))
        self.vis_pc(noised_one, 'hand/initial')

        # denoised 
        for i in range(3):
            noised_val = self.ema_model.p_sample(noised_val)        
            self.vis_pc(noised_val, 'nosied/step%d' % i)

        for i in range(3):
            noised_one = self.ema_model.p_sample(noised_one)        
            self.vis_pc(noised_one, 'hand/step%d' % i)
        return 
    
    def vis_pc(self, x, name):
        hand, obj = torch.chunk(x, 2, 1)
        hand = mesh_utils.pc_to_cubic_meshes(hand[..., :3], hand[..., 3:6])
        obj = mesh_utils.pc_to_cubic_meshes(obj[..., :3], obj[..., 3:6])
        hoi = mesh_utils.join_scene([hand, obj])
        image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
        self.logger.add_gifs(image_list, name, self.step)

    def set_input(self, data):
        pc = torch.cat([data['nHand_pc'].to(self.device), data['nObj_pc'].to(self.device)], 1)
        inputs = {'x': pc}
        return inputs


def main_function(gpu=None, ngpus_per_node=None, args=None):
    init_env(args, gpu, ngpus_per_node)
    rank = get_rank()
    world_size = get_world_size()

    device = torch.device('cuda', get_local_rank())

    exp_dir = args.exp_dir
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring=args.logging.get('mode', 'wandb'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=rank, is_master=is_master(), multi_process_logging=(world_size > 1))

    denoised = build_model(args)

    denoised.to(device)
    
    dataset, valset = build_dataset(args)
    print('dataset', len(dataset), len(valset))

    trainer = build_trainer(args, denoised, dataset, valset, logger)
    trainer.load(None, map_location=device)
    trainer.train()


def build_dataset(args):
    from ddpm.data import PCData
    dataset = PCData(args.train_split, data_dir=args.data_dir, train=True, args=args)
    valset = PCData(args.test_split, data_dir=args.data_dir, train=False, args=args)
    return dataset, valset


def build_trainer(args, denoised, dataset, valset, logger):
        trainer = Trainer(
            denoised,
            dataset,
            valset,
            start_time=args.time.start,
            device_ids=args.device_ids,
            train_batch_size = args.batch_size,
            train_lr = 2e-5,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = False,                        # turn on mixed precision
            args = args,
            logger = logger,
            results_folder=osp.join(args.exp_dir, 'ckpt'),
            save_and_sample_every=args.n_save_freq
        )        
        return trainer    


def build_model(args):
    model = PointNet2ClassificationSSG(**args.unet_config.params)

    diffusion = GaussianPoseDiffusion(
        model,
        image_size = args.point_reso,  # point size
        loss_type = 'l2',    # L1 or L2
        xyz_std=args.xyz_std,
        theta_std=args.theta_std,
    )
    return diffusion


def load_diffusion_model(ckpt_file: str):
    args_file = ckpt_file.split('/ckpt/')[0] + '/config.yaml'
    args = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.load(args_file), resolve=True)
    )
        
    diffusion = build_model(args)
    state_dict = torch.load(ckpt_file)

    model_utils.load_my_state_dict(diffusion, state_dict['ema'])
    return diffusion
