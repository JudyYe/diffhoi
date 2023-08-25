# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Created on Sat Sep 17 2022
# --------------------------------------------------------
import importlib
import wandb
import logging
import os
import os.path as osp
import shutil
from glide_text2im.text2im_model import Text2ImUNet
from glide_text2im.respace import SpacedDiffusion
import torch
import torchvision.utils as vutils

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from hydra import main
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from ddpm2d.utils import glide_util
from ddpm2d.utils.logger import LoggerCallback, build_logger
from ddpm2d.dataset.dataset import build_dataloader
from ddpm2d.utils.train_util import load_from_checkpoint
from jutils import model_utils

class BaseModule(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.template_size = None
        # self.glide_model, self.diffusion, self.glide_options = build_glide_model(cfg)
        self.glide_model: Text2ImUNet = None
        self.diffusion: SpacedDiffusion = None
        self.glide_options: dict = {}
        
        self.val_batch = None
        self.train_batch = None
        self.log_dir = osp.join(cfg.exp_dir, 'log')
    
    def train_dataloader(self):
        cfg = self.cfg
        dataloader = build_dataloader(cfg, cfg.trainsets, 
            self.glide_model.tokenizer, self.glide_options["text_ctx"],
            True, cfg.batch_size, True)
        for data in dataloader:
            self.train_batch = data
            break
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
    
    def val_dataloader(self):
        cfg =self.cfg
        val_dataloader = build_dataloader(cfg,  cfg.testsets, # cfg.data.test_data_dir, 
            self.glide_model.tokenizer, self.glide_options["text_ctx"],
            # cfg.data.test_split, 
            False, cfg.test_batch_size, False)
        for data in val_dataloader:
            self.val_batch = data
            break
        return val_dataloader

    def init_model(self):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [x for x in self.glide_model.parameters() if x.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.adam_weight_decay,
        )
        return optimizer

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        return

    def training_step(self, batch, batch_idx):
        self.train_batch = batch
        loss, losses = self.step(batch, batch_idx)
        if torch.isnan(loss):
            import pdb; pdb.set_trace()

        if self.global_step % self.cfg.print_frequency == 0:
            losses["loss"] = loss
            self.logger.log_metrics(losses, step=self.global_step)
        
            print('[%05d]: %f' % (self.global_step, loss))
            for k, v in losses.items():
                print('\t %08s: %f' % (k, v.item()))
        return loss 
    
    def validation_step(self, batch, batch_idx):
        val_batch = model_utils.to_cuda(self.val_batch, self.device)
        train_batch = self.train_batch

        log = {}
        self.vis_input(val_batch, 'val_input/', log)
        self.generate_sample_step(val_batch, 'val/', log)

        if train_batch is not None:
            self.vis_input(train_batch, 'train_input/', log)
            self.generate_sample_step(train_batch, 'train/', log)
        self.logger.log_metrics(log, self.global_step)

    def forward(self, batch, **kwargs):
        cfg = self.cfg 
        samples, sample_list = glide_util.sample(
            glide_model=self.glide_model,
            glide_options=self.glide_options,
            size=self.template_size,
            batch_size=len(batch['image']),
            guidance_scale=cfg.test_guidance_scale,
            device=self.device,
            prediction_respacing=cfg.sample_respacing,
            image_to_upsample=None,
            val_batch=batch,
            uncond_image=cfg.get('uncond_image', False),
            **kwargs,
        )
        return samples, sample_list

    def generate_sample_step(self, batch, pref, log, step=None, S=2):
        cfg = self.cfg 
        print('device', self.device)
        if step is None: step = self.global_step
        file_list = []
        step = self.global_step
        for k, v in batch.items():
            print(k, v.shape)
        for n in range(S):
            samples, sample_list = glide_util.sample(
                glide_model=self.glide_model,
                glide_options=self.glide_options,
                size=self.template_size,
                batch_size=len(batch['image']),
                guidance_scale=cfg.test_guidance_scale,
                device=self.device,
                prediction_respacing=cfg.sample_respacing,
                image_to_upsample=None,
                val_batch=batch,
                uncond_image=cfg.get('uncond_image', False),
            )
            self.vis_samples(batch, samples, sample_list, pref + '%d_' % n, log, step)
        return file_list
    
    def blend(self, mask, bg, r=0.75):
        overlay = mask * (bg * (1-r) + r) + (1 - mask) * bg
        return overlay

    def distribute_weight(self, grad, w, *args, **kwargs):
        return grad * w
    
    # @classmethod
    def decode_samples(self, tensor):
        return {
            'semantics': tensor, 
        }

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        
        log['%ssample' % pref] = wandb.Image(vutils.make_grid(samples))        
        return 

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        log["%sgt" % pref] = wandb.Image(vutils.make_grid(batch['image']))


def main_function(gpu=None, ngpus_per_node=None, args=None):
    """we need this just because of bad engine?"""
    main_worker(args)


# second stage
@main('../configs', 'sem_glide')   
def main_worker(cfg):
    # handle learning rate 
    print('main worker')
    # torch.backends.cudnn.benchmark = True
    if cfg.ndim is None:
        if cfg.mode.cond == 1: 
            cfg.ndim_cond = cfg.mode.mask + 3 * cfg.mode.normal + cfg.mode.depth + 2 * cfg.mode.uv
            cfg.ndim = cfg.mode.mask + 3 * cfg.mode.normal + cfg.mode.depth
        elif cfg.mode.cond == 0:  # geom 
            cfg.ndim_cond = 0
            cfg.ndim = cfg.mode.mask * 3 +  (3 * cfg.mode.normal + cfg.mode.depth) * 2
        elif cfg.mode.cond == -1:
            cfg.ndim_cond = 0
            cfg.ndim = cfg.mode.mask + 3 * cfg.mode.normal + cfg.mode.depth
    module = importlib.import_module(cfg.model.module)
    model_cls = getattr(module, cfg.model.model)
    model = model_cls(cfg, )
    model.init_model()
    model.cuda()

    # instantiate model
    if cfg.eval:
        trainer = pl.Trainer(gpus=-1,
                             default_root_dir=cfg.exp_dir,
                             )
        model = load_from_checkpoint(cfg.ckpt)
        print(cfg.exp_dir, cfg.ckpt)
        cfg.outputs_dir = osp.join(cfg.exp_dir, cfg.test_name)
        model.log_dir = cfg.outputs_dir
        model.freeze()
        trainer.test(model=model, verbose=False)
    else:
        os.makedirs(cfg.exp_dir, exist_ok=True)
        with open(osp.join(cfg.exp_dir, 'config.yaml', ), 'w') as fp:
            # OmegaConf.save()
            OmegaConf.save(cfg, fp, True)
        
        logger = build_logger(cfg)

        checkpoint_callback = ModelCheckpoint(
            monitor='step',
            save_top_k=cfg.save_topk,
            mode="max",
            every_n_train_steps=cfg.save_frequency,
            save_last=True,
            dirpath=osp.join(cfg.exp_dir, 'checkpoints'),
            filename='glide-ft-{step}'
        )

        # max_epoch = cfg.training.epoch

        val_kwargs = {}        
        if len(model.train_dataloader()) <cfg.log_frequency:
            val_kwargs['check_val_every_n_epoch'] = int(cfg.log_frequency) // len(model.train_dataloader())
        else:
            val_kwargs['val_check_interval'] = cfg.log_frequency
        model_summary = ModelSummary(2)
        trainer = pl.Trainer(
                             gpus=-1,
                             strategy='ddp',
                             num_sanity_val_steps=cfg.sanity_step,
                             limit_val_batches=1,
                             default_root_dir=cfg.exp_dir,
                             logger=logger,
                            #  max_epochs=max_epoch,
                             max_steps=cfg.max_steps,
                             callbacks=[model_summary, checkpoint_callback, LoggerCallback()],
                             gradient_clip_val=cfg.model.grad_clip,
                             gradient_clip_algorithm='norm',
                             **val_kwargs,
                             )
        ckpt_path = cfg.get('resume_train_from', None)
        if not osp.exists(ckpt_path):
            ckpt_path = None
        trainer.fit(model, ckpt_path=ckpt_path)
        
    return model

if __name__ == '__main__':
    main_worker()