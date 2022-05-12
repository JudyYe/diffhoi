from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import yaml
from ddpm.openai import UNetModel, default
from utils import dist_util, io_util
from utils.dist_util import get_local_rank, get_rank, get_world_size, init_env, is_master
from .ddpm import GaussianDiffusion
import os
import os.path as osp
from utils.logger import Logger
from jutils import model_utils

def main_function(gpu=None, ngpus_per_node=None, args=None):
    init_env(args, gpu, ngpus_per_node)
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    device = torch.device('cuda', get_local_rank())

    exp_dir = args.exp_dir
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring=args.logging.get('mode', 'wandb'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=rank, is_master=is_master(), multi_process_logging=(world_size > 1))

    diffusion = build_model(args)

    diffusion.to(device)
    
    dataset, valset = build_dataset(args)
    print('dataset', len(dataset), len(valset))

    trainer = build_trainer(args, diffusion, dataset, valset, logger)
    trainer.load(None, map_location=device)
    dist.barrier()
    trainer.train()


def build_trainer(args, diffusion, dataset, valset, logger):
    if args.data_mode == 'sdf32':
        from .ddpm import Trainer
        trainer = Trainer(
            diffusion,
            dataset,
            valset,
            start_time=args.time.start,
            device_ids=args.device_ids,
            train_batch_size = args.batch_size,
            train_lr = 2e-5,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                        # turn on mixed precision
            args = args,
            logger = logger,
            results_folder=osp.join(args.exp_dir, 'ckpt'),
            save_and_sample_every=args.n_save_freq
        )
    elif args.data_mode == 'pc':
        from .ddpm_pose import Trainer        
        trainer = Trainer(
            diffusion,
            dataset,
            valset,
            start_time=args.time.start,
            device_ids=args.device_ids,
            train_batch_size = args.batch_size,
            train_lr = 2e-5,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                        # turn on mixed precision
            args = args,
            logger = logger,
            results_folder=osp.join(args.exp_dir, 'ckpt'),
            save_and_sample_every=args.n_save_freq
        )        

    return trainer
        
def build_dataset(args):
    if args.data_mode == 'sdf32':
        from ddpm.data import SdfData
        dataset = SdfData(args.train_split, data_dir=args.data_dir)
        valset = SdfData(args.test_split, data_dir=args.data_dir)
    elif args.data_mode == 'pc':
        from ddpm.pc_data import PCData
        dataset = PCData(args.train_split, data_dir=args.data_dir)
        valset = PCData(args.test_split, data_dir=args.data_dir)
    return dataset, valset


def build_model(args):
    if args.unet_config.target == 'ddpm.openai.UNetModel':
        model = UNetModel(**args.unet_config.params)
    elif args.unet_config.target == 'uncond':
        from tutorial import Unet
        model = Unet(
            dim = 64,
            channels=1,
            dim_mults = (1, 2, 4, 8),
        )
    else:
        raise NotImplementedError(args.unet_config.target)

    diffusion = GaussianDiffusion(
        model,
        image_size = args.point_reso,  # point size
        mode=args.unet_config.mode,
        art_para=args.unet_config.get('art_para', None),
        channels=1,
        starttime=args.time.start,
        timesteps = args.time.total,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )
    print(model)
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
