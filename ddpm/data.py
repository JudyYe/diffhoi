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



class SdfData(Dataset):
    def __init__(self, split, data_dir='/glusterfs/yufeiy2/fair/data/obman/', args=dict()) -> None:
        super().__init__()
        self.sdf_dir = osp.join(data_dir, 'grid_sdf', '{}.npz')
        self.index_list = [line.strip() for line in open(osp.join(data_dir, '%s.txt' % split))]

        if osp.exists(osp.join(data_dir, 'center20.npy')):
            special_hA = np.load(osp.join(data_dir, 'center20.npy'))[:4]
            self.special_hA = torch.FloatTensor(special_hA)
        else:
            logging.warn('no %s/center20.npy' % data_dir)

    def __len__(self, ):
        return len(self.index_list)

    def __getitem__(self, index):
        """
        Return: 
            nSdf: (1, D, H, W)
            hA: (45, )
        """
        obj = np.load(self.sdf_dir.format(self.index_list[index]))
        
        sample = {}
        sample['nSdf'] = obj['nSdf'][None].astype(np.float32)
        sample['hA'] = obj['hA'].astype(np.float32)
        return sample


class PCData(Dataset):
    def __init__(self, split, data_dir='/glusterfs/yufeiy2/fair/data/obman/', train=False, args=dict()) -> None:
        super().__init__()
        self.train = train
        self.args = args
        self.index_list = [line.strip() for line in open(osp.join(data_dir, '%s.txt' % split))]
        self.data_dir = data_dir
        self.shape_dir = osp.join(args.shape_dir, '{}', '{}',  'models/model_normalized.obj')
        self.num_points = args.point_reso

        self.cache = True
        self.split = split
        if 'mode' in split:
            folder = 'train'
        else:
            folder = 'train' if 'train' in split else 'evaluation'
        self.meta_dir = os.path.join(self.data_dir, folder, 'meta_plus', '{}.pkl')
        self.cache_file = osp.join(self.data_dir, 'Cache', '%s_%s.pkl' % ('obman', self.split))
        self.cache_mesh = osp.join(self.data_dir, 'Cache', '%s_%s_mesh.pkl' % ('obman', self.split))

        if osp.exists(osp.join(data_dir, 'center20.npy')):
            special_hA = np.load(osp.join(data_dir, 'center20.npy'))[:4]
            self.special_hA = torch.FloatTensor(special_hA)
        else:
            logging.warn('no %s/center20.npy' % data_dir)
        
        self.hand_wrapper = ManopthWrapper().to('cpu')
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],
            'hTo': [],
        }
        self.preload_anno()

    def preload_anno(self):
        if osp.exists(self.cache_file):
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

    def __len__(self, ):
        return len(self.index_list)

    def load_mesh(self, ind):
        index = self.anno['cad_index'][ind]
        mesh = self.obj2mesh[index]
        return mesh

    def __getitem__(self, index):
        oMesh = self.load_mesh(index)
        hA = torch.FloatTensor(self.anno['hA'][index])[None]
        hHand, hJoints = self.hand_wrapper(None, hA)
        nTh = get_nTh(center=hJoints[:, 5])
        hTo = torch.FloatTensor(self.anno['hTo'][index])[None]
        nMesh = mesh_utils.apply_transform(oMesh, nTh @ hTo)
        nHand = mesh_utils.apply_transform(hHand, nTh)

        hHand_pc = self.sample_pts(nHand, 0)
        hMesh_pc = self.sample_pts(nMesh, 1)
        
        # some dropout and some noise? 
        hHand_pc = self.rdn_xyz_noise(hHand_pc)
        hMesh_pc = self.rdn_xyz_noise(hMesh_pc)

        hHand_pc = self.rdn_dropout(hHand_pc)
        hMesh_pc = self.rdn_dropout(hMesh_pc)


        sample = {}        
        sample['nHand_pc'] = hHand_pc.astype(np.float32)
        sample['nObj_pc'] = hMesh_pc.astype(np.float32)
        sample['hA'] = self.anno['hA'][index].astype(np.float32)
        return sample

    def rdn_xyz_noise(self, points):
        """give iid Gaussian jitter to point xyz (P, 3) """
        if self.train:
            stdev = self.args.DB.JITTER_XYZ
            jitter = np.random.randn(points.shape[0], 3) * stdev
            points[:, 0:3] += jitter
        return points

    def rdn_dropout(self, pc):
        """(P, 3)"""
        if self.train:
            num_points = pc.shape[0]
            max_dropout_ratio = self.args.DB.MAX_DROPOUT
            dropout_ratio = np.random.random() * max_dropout_ratio
            drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
            drop_idx = drop_idx[drop_idx < num_points]
            if len(drop_idx) > 0:
                pc[drop_idx] = 0
        return pc


    def sample_pts(self, mesh, cls, num_class=3):
        """sample num_points from pointcloud"""
        xyz = op_3d.sample_points_from_meshes(mesh, self.num_points, return_textures=False)
        obj = xyz[0].detach().numpy()  # (P, 6)

        color = np.zeros([self.num_points, num_class])
        color[cls] = 1

        obj = np.concatenate([obj, color], -1)
        return obj

    


class SdfFly(Dataset):
    def __init__(self, split, data_dir='/glusterfs/yufeiy2/fair/mesh_sdf/', args=dict()) -> None:
        super().__init__()
        self.data = data_dir
        self.index_list = json.load(open(osp.join(data_dir, split + '_all.json')))[split]['all']
        self.sdf_dir = osp.join(data_dir, 'SdfSamples/obman/all/{}.npz')
        self.reso = 32
        
    def load_sdf(self, ind):
        cad_index = self.index_list[ind]
        npz_file = self.sdf_dir.format(cad_index)
        sdf = unpack_sdf_samples(npz_file, 32**3)
        # grid = mesh_utils.make_grid(32, 0.5, 'cpu', order='xyz').reshape(1, -1, 3)
        # _, idx, _ = op_3d.knn_points(grid, sdf[None,:, :3], return_nn=True)
        # idx = idx.squeeze(0).squeeze(-1)
        # h = self.reso
        # sdf_grid = sdf[idx, -1].reshape(1, h, h, h)
        # return sdf_grid
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
        sample['nSdf'] = oSdf
        sample['index'] = self.index_list[index]
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