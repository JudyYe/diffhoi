import logging
import os
import os.path as osp
from hydra import main
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataio.sdf import GenericSdfHand
from ihoi.net import DGCNN, get_embedder, Decoder

from utils import dist_util, io_util
from utils.dist_util import get_local_rank, get_rank, get_world_size, init_env, is_master
from utils.hand_utils import ManopthWrapper, get_nTh, transform_nPoints_to_js
from utils.logger import Logger
from jutils import mesh_utils


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self._dl = None
        self.hand_wrapper = ManopthWrapper()

        self.my_logger:Logger = None
        if is_master():
            io_util.save_config(cfg, os.path.join(cfg.exp_dir, 'config.yaml'))

    def step(self, batch):
        raise NotImplementedError

    def vis_step(self, batch, out):
        # vis_origin
        nP = batch['nSdf'].shape[1]
        nTh = get_nTh(hA=batch['hA'], hand_wrapper=self.hand_wrapper)
        nHand, _ = self.hand_wrapper(nTh, batch['hA'])
        nNeg = batch['nSdf'][:, -100:, 0:3]
        nObj = mesh_utils.pc_to_cubic_meshes(nNeg)
        nHoi = mesh_utils.join_scene([nHand, nObj])
        # nHoi = nHand
        image_list = mesh_utils.render_geom_rot(nHoi, scale=True)
        self.my_logger.add_gifs(image_list, 'input/nHoi', self.global_step)        

        # vis_recon by hand 
        N = len(batch['nSdf'])
        sdf = lambda x: -self(x, batch['hA'], out.get('z', None)) + 0.5
        nObj = mesh_utils.batch_sdf_to_meshes(sdf, N, )
        nHoi = mesh_utils.join_scene([nHand, nObj])
        image_list = mesh_utils.render_geom_rot(nHoi, scale=True)
        self.my_logger.add_gifs(image_list, 'pred/nHoi', self.global_step)        
        return 

    def training_step(self, batch, batch_idx):
        loss, losses, out = self.step(batch)

        self.my_logger.log_metrics(
            {f"train/{k}": v for k, v in losses.items()}, step=self.global_step)
        self.my_logger.log_metrics(
            {f"train/total_loss": loss}, step=self.global_step)
        
        if self.global_rank == 0 and self.global_step % self.cfg.training.n_print == 0:
            print('[Step %04d] %.4f %s ' % (self.global_step, loss, self.cfg.expname))
            for k,v in losses.items():
                print('\t%010s: %.4f' % (k, v))
        return loss

    def validation_step(self, batch, batch_idx, *args):
        with torch.no_grad():
            loss, losses, out = self.step(batch)

        self.my_logger.log_metrics(
            {f"val/{k}": v for k, v in losses.items()}, step=self.global_step)
        self.my_logger.log_metrics(
            {f"val/total_loss": loss}, step=self.global_step)
    
        self.vis_step(batch, out, )
        return loss, losses, out

    def train_dataloader(self):
        if self._dl is None:
            self._dl = build_dataloader(self.cfg.data.train, 'train', self.cfg,  True, self.cfg.training.batch_size)
        return self._dl

    def val_dataloader(self):
        return build_dataloader(self.cfg.data.val, 'val', self.cfg, 
            False, self.cfg.test.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return build_dataloader(self.cfg.data.test, 'test', self.cfg,
            False, self.cfg.test.batch_size, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)


class GeoDGCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mode = cfg.enc.mode
        z_dim = cfg.z_dim
        self.hand_wrapper = ManopthWrapper()
        if cfg.enc.mode == 'geo':
            inp_dim = 0
        self.net = DGCNN(cfg.enc.params, z_dim*2, inp_dim)
    
    def get_pc(self, batch):
        if self.mode == 'geo':
            nTh = get_nTh(hA=batch['hA'], hand_wrapper=self.hand_wrapper)
            nHand, _ = self.hand_wrapper(nTh, batch['hA'])
            nObj = batch['nObj']  # (N, P, 3)
            nHoi = torch.cat([nHand.verts_padded(), nObj], 1)
        else:
            raise NotImplementedError
        return nHoi

    def forward(self, batch):
        xyz = self.get_pc(batch)
        z = self.net(xyz.transpose(-1, -2))
        mu, logvar = torch.chunk(z, 2, -1)
        return mu, logvar
    

class CondHandPrior(BaseModel):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.encoder = GeoDGCNN(cfg)
        self.decoder = HandPrior(cfg)

    def forward(self, nSdf, hA, z=None, *args, **kwargs):
        if z is None:
            mu, log_var = self.encoder(kwargs.get('batch'))
            z = mu
        return self.decoder(nSdf, hA, z, *args, **kwargs)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        if not self.cfg.resample:
            z = mu
        return p, q, z    

    def step(self, batch):
        mu, log_var = self.encoder(batch)
        
        p, q, z = self.sample(mu, log_var)
        
        cfg = self.cfg.loss
        # recon loss
        loss, losses, out = self.decoder.step(batch, z=z)  

        # kl loss
        kl = torch.distributions.kl_divergence(q, p)
        kl = cfg.kl * kl.mean()
        losses['kl'] = kl
        loss = loss + kl
        out['z'] = z

        return loss, losses, out

    def vis_step(self, batch, out):
        super().vis_step(batch, out)
        nTh = get_nTh(hA=batch['hA'], hand_wrapper=self.hand_wrapper)
        nHand, _ = self.hand_wrapper(nTh, batch['hA'])
        nObj = mesh_utils.pc_to_cubic_meshes(batch['nObj'])
        nHoi = mesh_utils.join_scene([nHand, nObj])
        # nHoi = nHand
        image_list = mesh_utils.render_geom_rot(nHoi, scale=True)
        self.my_logger.add_gifs(image_list, 'input/nPc', self.global_step)
        self.hallc_step(batch, out, nHand)

    def hallc_step(self, batch, out, nHand=None):
        S = 5
        z_dim = self.cfg.z_dim
        device = batch['nSdf'].device
        N = len(batch['nSdf'])

        z = torch.randn(S, z_dim, device=device)
        for i in range(len(z)):
            exp_z = z[i:i+1].repeat(N, 1)

            if nHand is None:
                nTh = get_nTh(hA=batch['hA'], hand_wrapper=self.hand_wrapper)
                nHand, _ = self.hand_wrapper(nTh, batch['hA'])

            sdf = lambda x: -self(x, batch['hA'], exp_z) + 0.5
            nObj = mesh_utils.batch_sdf_to_meshes(sdf, N, )
            nHoi = mesh_utils.join_scene([nHand, nObj])
            image_list = mesh_utils.render_geom_rot(nHoi, scale=True)
            self.my_logger.add_gifs(image_list, 'sample/nHoi%d' % i, self.global_step)        

        return 

class HandPrior(BaseModel):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        multires = cfg.deepsdf.pe_mul
        inp_dim = 17 
        if cfg.deepsdf.inp_mode == 'coord':
            inp_dim *= 3
        if not cfg.deepsdf.pe:
            multires = -1
        self.embed, out_dim = get_embedder(multires, inp_dim)
        self.occ_net = Decoder(**cfg.deepsdf.NetworkSpecs, xyz_dim=out_dim)

    def forward(self, nSdf, hA, z=None, *args, **kwargs):
        jsnPoints = self.bone_coordinate(nSdf, hA)
        jsnPoints = self.cat_z_points(z, jsnPoints)
        occ, occ_logit = self.occ_net(jsnPoints)
        return occ

    def bone_coordinate(self, nPoints, hA):
        N, P, _ = nPoints.size()
        jsPoints = transform_nPoints_to_js(self.hand_wrapper, hA, nPoints)  # (N ,P, J, 3)
        jsnPoints = torch.cat([jsPoints.reshape(N, P, -1), nPoints], -1)

        cfg = self.cfg.deepsdf
        assert not (cfg.inp_mode == 'coord' and cfg.inv), 'cannot inverse coordinate'
        # if distance or coordinate
        if cfg.inp_mode == 'dist':
            jsnPoints = torch.norm(jsnPoints.reshape(N, P, -1, 3), dim=-1)  # (N, P, J+1)
        # inverse or not
        if cfg.inv:
            jsnPoints = 1. / jsnPoints.clamp(min=cfg.inv_th)  # inv_th~ 0.001 (max 1000)
        # add pe or not 
        if cfg.pe:
            jsnPoints = self.embed(jsnPoints) 
        return jsnPoints  # (N, P, J*3+3)

    def cat_z_points(self, z, pts):
        """
        :param: z: (N, D)
        :param: pts: (N, P, Dp)
        """
        if z is None:
            return pts
        exp_z = repeat(z, 'N D -> N P D', P=pts.shape[1])
        return torch.cat([exp_z, pts], -1)

    def step(self, batch, z=None):
        nXyz = batch['nSdf'][..., 0:3]  # (N, P, 3)

        jsnPoints = self.bone_coordinate(nXyz, batch['hA'])  # (N ,P, (1+J) * 3 / 1)
        jsnPoints = self.cat_z_points(z, jsnPoints)
        
        occ, occ_logit = self.occ_net(jsnPoints)  # (N, P, 1)

        loss, losses, out = 0, {}, {}
        out['nOcc'] = occ
        out['nOcc_logit'] = occ_logit

        nSdf = batch['nSdf'][..., 3:4]  # [0 / 1]
        recon_loss = F.binary_cross_entropy_with_logits(occ_logit, nSdf)
        losses['recon'] = recon_loss
        loss = loss + recon_loss

        return loss, losses, out
    

def build_model(cfg):
    if cfg.mode == 'uncond':
        model = HandPrior(cfg)
    elif cfg.mode == 'cond':
        model = CondHandPrior(cfg)
    return model


def build_dataset(dset, split, train, cfg):
    if dset == 'obman':
        from dataio.obman import get_anno_split
        get_anno = get_anno_split(split=split)
    elif dset == 'mow':
        from dataio.mow import get_anno_split
        get_anno = get_anno_split(split=split)

    elif dset == 'ho3d': 
        from dataio.ho3d import get_anno_split
        get_anno = get_anno_split(split=split)
    dataset = GenericSdfHand(*get_anno, train=train, args=cfg)
    return dataset


def build_dataloader(dsets, split, cfg, is_train, bs, shuffle=None):
    dset_list = []
    for dset in dsets:
        dset_list.append(build_dataset(dset, split, is_train, cfg))
    if len(dset_list) > 1:
        dset = ConcatDataset(dset_list)
    else:
        dset = dset_list[0]
    if shuffle is None:
        shuffle = is_train

    if cfg.ddp:
        train_sampler = DistributedSampler(dset)
        dl = DataLoader(dset, sampler=train_sampler, 
                batch_size = bs, pin_memory=True, num_workers=cfg.environment.workers)
    else:
        dl = DataLoader(dset, batch_size = bs, 
            shuffle=shuffle, pin_memory=True, num_workers=cfg.environment.workers)
    return dl

@main(config_path='../configs', config_name='blind_prior')
def main(args):
    exp_dir = args.exp_dir
    logger = WandbLogger(
        name=osp.basename(exp_dir),
        save_dir=args.exp_dir,
        project='vhoi_%s' % osp.basename(osp.dirname(exp_dir)),
        log_model=True,
    )

    model = build_model(args)
    model.my_logger = logger
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.exp_dir,
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=args.n_save_freq,
    )
    
    trainer = pl.Trainer(
        strategy="ddp",
        devices="auto",
        # gpus=[local_rank],
        # gradient_clip_val=args.deepsdf.TrainSpecs.GradientClipNorm,
        num_sanity_val_steps=1,
        limit_val_batches=2,
        check_val_every_n_epoch=max(1, args.n_eval_freq // len(model.train_dataloader())),
        max_steps=args.max_step,
        default_root_dir=args.exp_dir,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint=ckpt,
        enable_progress_bar=is_master(),
    )

    trainer.fit(model)



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

    model = build_model(args)
    model.my_logger = logger
    if args.training.ckpt_file is not None:
        logging.info('loading from %s' % args.training.ckpt_file)
        state_dict = torch.load(args.training.ckpt_file)
        model.load_state_dict(state_dict['state_dict'])
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=args.n_save_freq,
    )
    # checkpoint_callback = model_utils.CheckpointEveryNSteps(args.n_save_freq, args.n_eval_freq, args.special_ckpt, args.special_ckpt)
    
    filepath = osp.join(args.exp_dir, 'ckpt', 'last.pth')
    ckpt = filepath if osp.exists(filepath) else None
    print(ckpt, filepath)

    trainer = pl.Trainer(
        strategy="ddp",
        gpus=[local_rank],
        # gradient_clip_val=args.deepsdf.TrainSpecs.GradientClipNorm,
        num_sanity_val_steps=1,
        limit_val_batches=2,
        check_val_every_n_epoch=max(1, args.n_eval_freq // len(model.train_dataloader())),
        max_steps=args.max_step,
        default_root_dir=args.exp_dir,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=ckpt,
        enable_progress_bar=is_master(),
    )
    if args.environment.multiprocessing_distributed:
        dist.barrier()

    trainer.fit(model)


if __name__ == '__main__':
    main()