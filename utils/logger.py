from time import sleep
import wandb
from utils import io_util
from utils import dist_util
from utils.dist_util import is_master
from utils.print_fn import log

import os
import os.path as osp
import torch
import pickle
import imageio
import torchvision
import numpy as np

import torch.distributed as dist
from jutils import image_utils
#---------------------------------------------------------------------------
#---------------------- tensorboard / image recorder -----------------------
#---------------------------------------------------------------------------

class Logger(object):
    """
    modified from https://github.com/LMescheder/GAN_stability/blob/master/gan_training/logger.py
    """
    def __init__(self,
                 log_dir,
                 img_dir,
                 monitoring=None,
                 monitoring_dir=None,
                 rank=0,
                 is_master=True,
                 multi_process_logging=True):
        self.stats = dict()
        self.log_dir = log_dir
        self.img_dir = img_dir
        self.rank = rank
        self.is_master = is_master
        self.multi_process_logging = multi_process_logging

        if self.is_master:
            io_util.cond_mkdir(self.log_dir)
            io_util.cond_mkdir(self.img_dir)
        if self.multi_process_logging:
            dist.barrier()

        self.monitoring = None
        self.monitoring_dir = None

        self.wandb = monitoring == 'wandb'
        # NOTE: for now, we are allowing tensorboard writting on all child processes, 
        #       as it's already nicely supported, 
        #       and the data of different events file of different processes will be automatically aggregated when visualizing.
        #       https://discuss.pytorch.org/t/using-tensorboard-with-distributeddataparallel/102555/7
        if not (monitoring is None or monitoring == 'none'):
            self.setup_monitoring(monitoring, monitoring_dir)


    def setup_monitoring(self, monitoring, monitoring_dir):
        self.monitoring = monitoring
        self.monitoring_dir = monitoring_dir
        if monitoring == 'tensorboard':
            # NOTE: since torch 1.2
            from torch.utils.tensorboard import SummaryWriter
            # from tensorboardX import SummaryWriter
            self.tb = SummaryWriter(self.monitoring_dir)
        elif monitoring == 'wandb':
            if self.is_master:
                log_dir = self.log_dir
                os.makedirs(os.path.join(log_dir, 'wandb'), exist_ok=True)
                wandb.init(
                    project='vhoi_%s' % log_dir.split('/')[-2],
                    name='/'.join(log_dir.split('/')[-2:]),
                    dir=log_dir,
                    entity='judy_smith',
                    resume="allow",
                    # id='_'.join(log_dir.split('/')[-2:]),
                )
        else:
            raise print('Monitoring tool "%s" not supported!'
                                      % monitoring)

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

        k_name = '%s/%s' % (category, k)
        if self.monitoring == 'telemetry':
            self.tm.metric_push_async({
                'metric': k_name, 'value': v, 'it': it
            })
        elif self.monitoring == 'tensorboard':
            self.tb.add_scalar(k_name, v, it)

        if is_master() and self.wandb:
            wandb.log({k_name: v}, step=it)

    def add_vector(self, category, k, vec, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        if isinstance(vec, torch.Tensor):
            vec = vec.data.clone().cpu().numpy()

        self.stats[category][k].append((it, vec))

    def add_gif_files(self, file_path, class_name, it):
        if is_master() and self.wandb:
            wandb.log({class_name: wandb.Video(file_path)}, step=it)

    def add_gifs(self,image_list, class_name, it):
        outdir = os.path.join(self.img_dir, class_name)
        if self.is_master and not os.path.exists(outdir):
            os.makedirs(outdir)
        if self.multi_process_logging:
            dist.barrier()

        outfile = os.path.join(outdir, '{:08d}_{}'.format(it, self.rank))

        # imgs = imgs / 2 + 0.5
        image_utils.save_gif(image_list, outfile)
        
        if is_master() and self.wandb:
            wandb.log({class_name: wandb.Video(outfile + '.gif')}, step=it)

        
    def add_imgs(self, imgs, class_name, it):
        outdir = os.path.join(self.img_dir, class_name)
        outfile = os.path.join(outdir, '{:08d}_{}.png'.format(it, self.rank))
        if self.is_master:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
        if self.multi_process_logging:
            dist.barrier()

        # imgs = imgs / 2 + 0.5
        imgs = torchvision.utils.make_grid(imgs)
        torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)
        if is_master() and self.wandb:
            wandb.log({class_name: wandb.Image(imgs)}, step=it)

        if self.monitoring == 'tensorboard':
            self.tb.add_image(class_name, imgs, global_step=it)

    def add_meshes(self, name, mesh_file, it):
        if is_master() and self.wandb and not mesh_file.endswith('.ply'):
            wandb.log({name: wandb.Object3D(open(osp.join(self.log_dir, mesh_file)))}, step=it)

    def add_figure(self, fig, class_name, it, save_img=True):
        if save_img:
            outdir = os.path.join(self.img_dir, class_name)
            if self.is_master and not os.path.exists(outdir):
                os.makedirs(outdir)
            if self.multi_process_logging:
                dist.barrier()
            outfile = os.path.join(outdir, '{:08d}_{}.png'.format(it, self.rank))

            image_hwc = io_util.figure_to_image(fig)
            imageio.imwrite(outfile, image_hwc)
            if self.monitoring == 'tensorboard':
                if len(image_hwc.shape) == 3:
                    image_hwc = np.array(image_hwc[None, ...])
                self.tb.add_images(class_name, torch.from_numpy(image_hwc), dataformats='NHWC', global_step=it)
        else:
            if self.monitoring == 'tensorboard':
                self.tb.add_figure(class_name, fig, it)

    def add_module_param(self, module_name, module, it):
        if self.monitoring == 'tensorboard':
            for name, param in module.named_parameters():
                self.tb.add_histogram("{}/{}".format(module_name, name), param.detach(), it)

    def get_last(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]
    
    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename + '_{}'.format(self.rank))
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename + '_{}'.format(self.rank))
        if not os.path.exists(filename):
            # log.info('=> File "%s" does not exist, will create new after calling save_stats()' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
                log.info("=> Load file: {}".format(filename))
        except EOFError:
            log.info('Warning: log file corrupted!')
