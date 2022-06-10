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


class SdfHand(Dataset):
    """SDF Wrapper of datasets"""
    def __init__(self, split, data_dir='/glusterfs/yufeiy2/fair/data/obman', train=False, args=dict(), base_idx=0):
        super().__init__()
        self.cfg = args
        dataset = 'obman'
        self.dataset = dataset
        self.train = train
        self.split = split
        
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],   # torch.Float (45? )
            'hTo': [],  # torch Float (4, 4)
        }

        self.base_idx = base_idx
        self.data_dir = data_dir

        self.cache = True
        if 'mode' in split:
            folder = 'train'
        else:
            folder = 'train' if 'train' in split else 'evaluation'

        self.cache_file = osp.join(self.data_dir, 'Cache', '%s_%s.pkl' % (dataset, self.split))
        self.cache_mesh = osp.join(self.data_dir, 'Cache', '%s_%s_mesh.pkl' % (dataset, self.split))

        self.shape_dir = os.path.join(self.cfg.shape_dir,  '{}', '{}', 'models', 'model_normalized.obj')
        self.meta_dir = os.path.join(self.data_dir, folder, 'meta_plus', '{}.pkl')

        self.subsample = args.point_reso

        self.hand_wrapper = ManopthWrapper().to('cpu')

        self.preload_anno()

    def preload_anno(self, load_keys=[]):
        
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!')
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:

            index_list = [line.strip() for line in open(osp.join(self.data_dir, '%s.txt' % self.split))]
            for i, index in enumerate(tqdm.tqdm(index_list)):

                meta_path = self.meta_dir.format(index)
                with open(meta_path, "rb") as meta_f:
                    meta_info = pickle.load(meta_f)

                self.anno['index'].append(index)
                self.anno['cad_index'].append(osp.join(meta_info["class_id"], meta_info["sample_id"]))
                cTo = torch.FloatTensor([meta_info['cTo']])
                cTh = torch.FloatTensor([meta_info['cTh']])
                hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True)
                hTo = torch.matmul(hTc, cTo)

                self.anno['hTo'].append(hTo[0])
                self.anno['hA'].append(meta_info['hA'])

            os.makedirs(osp.dirname(self.cache_file), exist_ok=True)
            print('save cache')
            pickle.dump(self.anno, open(self.cache_file, 'wb'))

        self.preload_mesh()

    def preload_mesh(self):
        if self.cache and osp.exists(self.cache_mesh):
            print('!! Load from cache !!')
            self.obj2mesh = pickle.load(open(self.cache_mesh, 'rb'))
        else:
            self.obj2mesh = {}
            print('load mesh')
            for i, cls_id in tqdm.tqdm(enumerate(self.anno['cad_index']), total=len(self.anno['cad_index'])):
                key = cls_id
                cls, id = key.split('/')
                if key not in self.obj2mesh:
                    fname = self.shape_dir.format(cls, id)
                    self.obj2mesh[key] = mesh_utils.load_mesh(fname, scale_verts=1)
            print('save cache')
            pickle.dump(self.obj2mesh, open(self.cache_mesh, 'wb'))

    def __len__(self):
        return len(self.anno['index'])

    def __getitem__(self, idx):
        sample = {}
        idx = self.map[idx] if self.map is not None else idx
        # load SDF
        cad_idx = self.anno['cad_index'][idx]
        filename = self.dataset.get_sdf_files(cad_idx)

        oPos_sdf, oNeg_sdf = unpack_sdf_samples(filename, None)
        hTo = torch.FloatTensor(self.anno['hTo'][idx])
        hA = torch.FloatTensor(self.anno['hA'][idx])
        nTh = get_nTh(self.hand_wrapper, hA[None], self.cfg.DB.RADIUS)[0]

        nPos_sdf = self.norm_points_sdf(oPos_sdf, nTh @ hTo) 
        nNeg_sdf = self.norm_points_sdf(oNeg_sdf, nTh @ hTo) 

        oSdf = torch.cat([
                self.sample_points(oPos_sdf, self.subsample),
                self.sample_points(oNeg_sdf, self.subsample),
            ], dim=0)
        sample['oSdf'] = oSdf

        nPos_sdf = self.sample_unit_cube(nPos_sdf, self.subsample)
        nNeg_sdf = self.sample_unit_cube(nNeg_sdf, self.subsample)
        nSdf = torch.cat([nPos_sdf, nNeg_sdf], dim=0)
        sample['nSdf'] = nSdf

        sample['hA'] = hA

        sample['indices'] = idx + self.base_idx
        sample['index'] = self.get_index(idx)
        return sample   

    def norm_points_sdf(self, obj, nTh):
        """
        :param obj: (P, 4)
        :param nTh: (4, 4)
        :return:
        """
        D = 4

        xyz, sdf = obj[None].split([3, D - 3], dim=-1)  # (N, Q, 3)
        nXyz = mesh_utils.apply_transform(xyz, nTh[None])  # (N, Q, 3)
        _, _, scale = geom_utils.homo_to_rt(nTh)  # (N, 3)
        # print(scale)  # only (5 or 1???)
        sdf = sdf * scale[..., 0:1, None]  # (N, Q, 1) -> (N, 3)
        nObj = torch.cat([nXyz, sdf], dim=-1)
        return nObj[0]

    def sample_points(self, points, num_points):
        """
        Args:
            points ([type]): (P, D)
        Returns:
            sampled points: (num_points, D)
        """
        P, D = points.size()
        ones = torch.ones([P])
        inds = torch.multinomial(ones, num_points, replacement=True).unsqueeze(-1)  # (P, 1)
        points = torch.gather(points, 0, inds.repeat(1, D))
        return points

    def sample_unit_cube(self, hObj, num_points, r=1):
        """
        Args:
            points (P, 4): Description
            num_points ( ): Description
            r (int, optional): Description
        
        Returns:
            sampled points: (num_points, 4)
        """
        D = hObj.size(-1)
        points = hObj[..., :3]
        prob = (torch.sum((torch.abs(points) < r), dim=-1) == 3).float()
        if prob.sum() == 0:
            prob = prob + 1
            print('oops')
        inds = torch.multinomial(prob, num_points, replacement=True).unsqueeze(-1)  # (P, 1)

        handle = torch.gather(hObj, 0, inds.repeat(1, D))
        return handle

    def get_index(self, idx):
        index =  self.anno['index'][idx]
        if isinstance(index, tuple) or isinstance(index, list):

            index = '/'.join(index)
        return index




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