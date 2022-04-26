from .ddpm import Trainer, GaussianDiffusion, Unet
import os
from utils.logger import Logger


def main(args):
    exp_dir = args.exp_dir
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring=args.training.get('monitoring', 'wandb'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=0, is_master=True, multi_process_logging=False)

    model = Unet(
        dim = 64,
        channels=1,
        dim_mults = (1, 2, 4, 8),
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = args.point_reso,  # point size
        channels=1,
        timesteps = 100,   # number of steps
        loss_type = 'l2'    # L1 or L2
    ).cuda()

    dataset = SdfData(args)
    valset = SdfData(args, grid=True)
    trainer = Trainer(
        diffusion,
        dataset,
        valset,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                        # turn on mixed precision
        args = args,
        logger = logger,
    )

    trainer.train()    

