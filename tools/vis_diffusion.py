"""script to debug SDLoss"""
import numpy as np
import os
import os.path as osp
from time import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from models.frameworks.volsdf_hoi import VolSDFHoi, Trainer
from preprocess.smooth_hand import vis_hA, vis_se3
from utils import io_util, mesh_util
from utils.dist_util import is_master
from glob import glob
from models.frameworks import get_model
from models.cameras import get_camera
from dataio import get_data

import torch
from torch.utils.data.dataloader import DataLoader
from jutils import image_utils, geom_utils, mesh_utils, model_utils

from models.sd import SDLoss

def get_inp(image_file):
    # TODO: emmm is GLIDe trained with scale [-1, 1] or [0, 1?]-->[-1, 1]
    image = Image.open(image_file)
    H = W = args.reso
    image = image.resize((H, W))
    image = ToTensor()(image)
    image = image * 2 - 1
    return 


def main_function(args):
    save_dir = args.out
    sd = SDLoss(args.load_pt)
    sd.init_model()

    image_list = sorted(glob(osp.join(args.inp)))

    name_list = ['noisy_img', 'multi_step', 'single_step']
    image_list = [[] for _ in name_list]
    text_list = [[] for _ in name_list]
    for image_file in image_list:
        for noise_level in args.noise.split(','):
            image = get_inp(image_file)

            noisy = sd.get_noisy_image(image, noise_level)

            m_list, s_list = [], []
            for _ in args.S:
                multi_out = sd.vis_multi_step(noisy, noise_level)
                single_out = sd.vis_one_step(noisy, noise_level)
                m_list.append(multi_out)
                s_list.append(single_out)
            multi_out = torch.cat(m_list, dim=-2) # concat in height
            single_out = torch.cat(s_list, dim=-2)

            image_list[0].append(noisy)
            image_list[1].append(multi_out)
            image_list[2].append(single_out)

            text_list[1].append(f'multi step: {noise_level:g}')
            text_list[2].append(f'single step: {noise_level:g}')
    image_utils.save_images(image, osp.join(save_dir, name + '_inp'), scale=True)
    for name, images, texts in zip(name_list, image_list, text_list):
        # [(1, C, H, W), ]
        if len(texts) == 0:
            texts.append(None)
        image_utils.save_images(
            torch.cat(images, dim=0),
            osp.join(save_dir, name),
            texts, col=len(images), scale=True
        )


        

    H = W = args.reso
    device = 'cuda:0'
    # load config
    config = io_util.load_yaml(osp.join(args.load_pt.split('/ckpts')[0], 'config.yaml'))

    # load data    
    dataset, _ = get_data(config, return_val=True, val_downscale=1)
    dataloader = DataLoader(dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mesh_utils.collate_meshes)

    # build and load model 
    posenet, focal_net = get_camera(config, datasize=len(dataset)+1, H=dataset.H, W=dataset.W)
    model, trainer, render_kwargs_train, render_kwargs_test, _, _ = get_model(config, data_size=len(dataset)+1, cam_norm=dataset.max_cam_norm, device=[0])

    assert args.out is not None or args.load_pt is not None, 'Need to specify one of out / load_pt'
    
    if args.load_pt is not None:
        state_dict = torch.load(args.load_pt)
        
        model.load_state_dict(state_dict['model'])
        posenet.load_state_dict(state_dict['posenet'])
        focal_net.load_state_dict(state_dict['focalnet'])
        
        trainer.init_camera(posenet, focal_net)
        trainer.to(device)
        trainer.eval()
        
        it = state_dict['global_step']
        name = 'it%08d' % it
        save_dir = args.load_pt.split('/ckpts')[0] + '/render/'
    
    if args.out is not None:
        save_dir = args.out

    os.makedirs(save_dir, exist_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help='output ply file name')
    parser.add_argument('--reso', type=int, default=224, help='resolution of images')
    parser.add_argument('--N', type=int, default=64, help='resolution of the marching cube algo')
    parser.add_argument('--D', type=int, default=128, help='resolution of the marching cube algo')
    parser.add_argument('--T', type=int, default=100, help='resolution of the marching cube algo')
    parser.add_argument('--volume_size', type=float, default=6., help='voxel size to run marching cube')

    parser.add_argument("--diff_ckpt", type=str, default='', help='the trained model checkpoint .pt file')
    parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    parser.add_argument("--chunk", type=int, default=4096*2, help='net chunk when querying the network. change for smaller GPU memory.')

    args = parser.parse_args()
    
    main_function(args)