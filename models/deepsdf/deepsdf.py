import logging
from pytorch_lightning.callbacks import ModelCheckpoint
import functools
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from importlib import import_module
from einops import rearrange, repeat
import math
from typing import Any
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import yaml
from dataio.sdf import Sdf
from ddpm.ddpm_pose import build_dataset
from utils import dist_util, io_util
from utils.dist_util import get_local_rank, get_rank, get_world_size, init_env, is_master
import os
import os.path as osp
from utils.logger import Logger
from jutils import model_utils, mesh_utils, image_utils


def exists(val):
    return val is not None
# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)

        # FiLM modulation

        if exists(gamma):
            out = out * gamma

        if exists(beta):
            out = out + beta

        out = self.activation(out)
        return out


# mapping network

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 0.1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class MappingNetwork(nn.Module):
    def __init__(self, *, dim, dim_out, depth = 3, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

        self.to_gamma = nn.Linear(dim, dim_out)
        self.to_beta = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)


# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)
        return self.last_layer(x)



class SirenDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_hidden,
        siren_num_layers = 8
    ):
        super().__init__()

        self.mapping = MappingNetwork(
            dim = dim,
            dim_out = dim_hidden
        )

        self.siren = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            num_layers = siren_num_layers
        )

        self.to_alpha = nn.Linear(dim_hidden, 1)


    def forward(self, latent, coors, batch_size = 8192):
        gamma, beta = self.mapping(latent)

        outs = []
        for coor in coors.split(batch_size):
            gamma_, beta_ = map(lambda t: rearrange(t, 'n -> () n'), (gamma, beta))
            x = self.siren(coor, gamma_, beta_)
            alpha = self.to_alpha(x)

            out = alpha
            outs.append(out)

        return torch.cat(outs)


# DeepSDF
class Decoder(pl.LightningModule):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()


        dims = [latent_size + 3] + list(dims) + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


class AutoDecoder(pl.LightningModule):
    def __init__(self, decoder_mode, surface_kwargs, train_kwargs, args, **kwargs):
        super().__init__()
        self.args = args
        if decoder_mode == 'models.deepsdf.deepsdf.Decoder':
            latent_size = train_kwargs.CodeLength
            self.decoder = Decoder(**surface_kwargs)
        self.embedding : nn.Embedding = None
        self.train_kwargs = train_kwargs
        self.minT, self.maxT = -train_kwargs.ClampingDistance, train_kwargs.ClampingDistance
        self.my_logger = None
        self.dl = None
        self.dl = self.train_dataloader()
        if is_master():
            io_util.save_config(args, os.path.join(args.exp_dir, 'config.yaml'))

    def train_dataloader(self):
        if self.dl is not None:
            return self.dl
        specs = self.train_kwargs
        latent_size = specs.CodeLength
        code_bound = specs.CodeBound

        args = self.args
        train_batch_size = args.batch_size
        dset = build_dataset(self.args, train=True)
        print('train', len(dset))
        if args.ddp:
            train_sampler = DistributedSampler(dset)
            dl = data.DataLoader(dset, sampler=train_sampler, 
                    batch_size = train_batch_size, pin_memory=True, num_workers=args.environment.workers)
        else:
            dl = data.DataLoader(self.ds, batch_size = train_batch_size, 
                shuffle=True, pin_memory=True, num_workers=args.environment.workers)

        num_scenes = len(dset)
        self.embedding = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound, device=self.device)
        self.dl = dl
        return dl

    def val_dataloader(self):
        args = self.args
        train_batch_size = args.batch_size
        dset = build_dataset(self.args, train=False)
        self.valset = dset
        print('val', len(dset))
        
        if args.ddp:
            train_sampler = DistributedSampler(dset)
            dl = data.DataLoader(dset, sampler=train_sampler, 
                    batch_size = train_batch_size, pin_memory=True, num_workers=args.environment.workers)
        else:
            dl = data.DataLoader(self.ds, batch_size = train_batch_size, 
                shuffle=False, pin_memory=True, num_workers=args.environment.workers)
        return dl        
        
    def training_step(self, batch, batch_idx):
        batch_split = 1
        num_samp_per_scene = self.train_kwargs['SamplesPerScene']
        enforce_minmax = self.train_kwargs['enforce_minmax']
        th = self.train_kwargs['ClampingDistance']
        minT, maxT = -th, th
        do_code_regularization = self.train_kwargs['CodeRegularization']
        code_reg_lambda = self.train_kwargs['CodeRegularizationLambda']

        lat_vecs = self.embedding
        decoder = self.decoder

        sdf_data = batch[self.args.frame]  # (B, P, 4)
        indices = batch['indices']

        # Process the input data
        sdf_data = sdf_data.reshape(-1, 4)

        num_sdf_samples = sdf_data.shape[0]

        sdf_data.requires_grad = False

        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, minT, maxT)

        xyz = torch.chunk(xyz, batch_split)
        indices = torch.chunk(
            indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
            batch_split,
        )

        sdf_gt = torch.chunk(sdf_gt, batch_split)

        batch_loss = 0.0

        # optimizer_all.zero_grad()

        for i in range(batch_split):

            batch_vecs = lat_vecs(indices[i])

            input = torch.cat([batch_vecs, xyz[i]], dim=1)

            # NN optimization
            pred_sdf = decoder(input)

            if enforce_minmax:
                pred_sdf = torch.clamp(pred_sdf, minT, maxT)

            chunk_loss = F.l1_loss(pred_sdf, sdf_gt[i].cuda(), reduction='sum') / num_sdf_samples

            if do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = (
                    code_reg_lambda * min(1, self.current_epoch / 100) * l2_size_loss
                ) / num_sdf_samples

                chunk_loss = chunk_loss + reg_loss.cuda()

            # chunk_loss.backward()

            batch_loss += chunk_loss

        loss = batch_loss

        if is_master() and self.global_step % 100 == 0:
            print(self.args.exp_dir)
            print('[%5d] loss: %f' % (self.global_step, loss.item()))
        self.my_logger.add('losses', 'total', loss.item(), self.global_step)

        return loss

    def training_step_end(self, *args, **kwargs):
        if self.global_step % self.args.n_save_freq == 0 or self.global_step in self.args.special_ckpt:
            self.trainer.save_checkpoint(
                osp.join(self.args.exp_dir, 'ckpt', 'iter%d.pth' % self.global_step
            ))
            self.trainer.save_checkpoint(
                osp.join(self.args.exp_dir, 'ckpt', 'last.pth'
            ))

        return super().training_step_end(*args, **kwargs)

    def sdf(self, x, latent):
        """
        :param x: (N, P, 3)
        :param latent: (N, D)? 
        :param output: (N, P, 1)
        """
        N, P, _3 = x.shape
        N, D = latent.shape
        latent = latent.unsqueeze(1).repeat(1, P, 1)
        inputs = torch.cat([latent, x], dim=-1).reshape(N * P, _3 + D)
        pred = self.decoder(inputs)
        pred = pred.reshape(N, P, 1)
        return pred

    def validation_step(self, batch, batch_idx):
        # vis
        # x: N, P, 3
        bs = self.args.batch_size
        D = self.args.deepsdf.TrainSpecs.CodeLength

        if self.embedding is not None:
            indices = torch.randint(0, len(self.valset), [bs], device=self.device)
            latent = self.embedding(indices)
            sdf = functools.partial(self.sdf, latent=latent)
            meshes = mesh_utils.batch_sdf_to_meshes(sdf, self.args.batch_size, )
            if not meshes.isempty():
                image_list = mesh_utils.render_geom_rot(meshes, scale_geom=True)
                self.my_logger.add_gifs(image_list, 'generate/embed', self.global_step)
            else: 
                logging.warn('empty mesh!!!')

        latent = torch.randn([bs, D], device=self.device)
        sdf = functools.partial(self.sdf, latent=latent)
        meshes = mesh_utils.batch_sdf_to_meshes(sdf, self.args.batch_size, )
        if not meshes.isempty():
            image_list = mesh_utils.render_geom_rot(meshes, scale_geom=True)
            self.my_logger.add_gifs(image_list, 'generate/gaussian', self.global_step)
        else:
            logging.warn('empty mesh')


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.train_kwargs.LearningRateSchedule.Initial)

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
        gpus=[gpu],
        gradient_clip_val=args.deepsdf.TrainSpecs.GradientClipNorm,
        num_sanity_val_steps=1,
        limit_val_batches=2,
        check_val_every_n_epoch=args.n_eval_freq // len(model.dl),
        max_steps=args.max_step,
        default_root_dir=args.exp_dir,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=ckpt,
        enable_progress_bar=is_master(),
    )
    dist.barrier()

    trainer.fit(model)


def build_model(args):
    sdf_arg = args.deepsdf
    if 'AutoDecoder' in sdf_arg.arch:
        model = AutoDecoder(sdf_arg.dec_arch, sdf_arg.NetworkSpecs, 
            sdf_arg.TrainSpecs, args)
    return model


def build_dataset(args, train=False):
    if args.data_mode == 'sdf':
        dataset = Sdf(args.train_split, data_dir=args.data_dir, train=train, args=args)
    return dataset