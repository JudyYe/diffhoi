import json
import logging
from time import time
import numpy as np
import os
import os.path as osp
import pickle
import tqdm

import pytorch3d.ops as op_3d

import torch
from torch.utils.data import Dataset
from jutils import geom_utils, mesh_utils
import yaml
from utils.hand_utils import ManopthWrapper, get_nTh


class Sdf(Dataset):
    def __init__(self, split, data_dir='/glusterfs/yufeiy2/fair/mesh_sdf/', train=False, args=dict()) -> None:
        super().__init__()
        self.data = data_dir
        self.train = train
        self.args = args
        self.index_list = json.load(open(osp.join(data_dir, split + '_all.json')))[split]['all']
        self.sdf_dir = osp.join(data_dir, 'SdfSamples/obman/all/{}.npz')
        self.num_points = args.point_reso
        
    def load_sdf(self, ind):
        cad_index = self.index_list[ind]
        npz_file = self.sdf_dir.format(cad_index)
        sdf = unpack_sdf_samples(npz_file, self.num_points)
        sdf[..., :3] *= 2
        return sdf
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        """
        Return: 
            nSdf: (1, D, H, W)
            hA: (45, )
        """
        oSdf = self.load_sdf(index)
        
        sample = {}
        sample['oSdf'] = oSdf
        sample['indices'] = index
        return sample



def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    if subsample is None:
        return torch.cat([pos_tensor, neg_tensor], 0)

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]





if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils import data
    from jutils import model_utils, image_utils
    save_dir = '/glusterfs/yufeiy2/vhoi/vis/'
    # args = OmegaConf.create(OmegaConf.to_container(OmegaConf.load('configs/ddpm_pose.yaml'), resolve=True))
    dataset = SdfFly('obman')
    torch.manual_seed(123)
    bs = 1
    dl = data.DataLoader(dataset, batch_size = bs, 
                shuffle=True, pin_memory=True, num_workers=10)
    device = 'cuda:0'
    for i, batch in enumerate(dl):
        print('one')
        t = time()
        batch = model_utils.to_cuda(batch, device)
        cmd = 'cp /glusterfs/yufeiy2/download_data/ShapeNetCore.v2/%s/models/model_normalized.obj %s' % (batch['index'][0], osp.join(save_dir, '%d_gt.obj' % i))
        os.system(cmd)
        print('grid')
        hoi = mesh_utils.batch_grid_to_meshes(batch['nSdf'].squeeze(1), bs)
        print('another grid', time() - t)
        # image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
        # image_utils.save_gif(image_list, osp.join(save_dir, '%d' % (i)))
        mesh_utils.dump_meshes(osp.join(save_dir, '%d' % i), hoi, )

        if i > 5:
            break