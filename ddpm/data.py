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
from trimesh.exchange.binvox import load_binvox
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
            nSdf: (P, 4)
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



# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

class Voxel(Dataset):
    def __init__(self, split, data_dir='/ShapeNet.Core.v2/', 
                 cats=['bottle', 'bowl', 'can', 'jar', 'knife', 'cellphone', 'camera', 'remote_control'], 
                 args=dict()) -> None:
        super().__init__()
        self.data = data_dir
        cats = [cate_to_synsetid[e] for e in cats]
        self.index_list = []
        for cat in cats:
            index_list = [osp.join(cat, line.strip()) for line in open(osp.join(data_dir, cat, split + '.txt'))]
            self.index_list += index_list
        np.random.seed(123)
        np.random.shuffle(self.index_list)
        
        self.vox_dir = osp.join(data_dir, '{}/models/model_normalized.solid.binvox')
        self.reso = 32
                
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        """
        Return: 
            nSdf: (1?, D, H, W)
            hA: (45, )
        """
        with open(self.vox_dir.format(self.index_list[index]), 'rb') as fp:
            vox = load_binvox(fp)

        vox128 = np.array(vox.matrix).astype(np.float32)
        H = self.reso
        vox32 = (np.resize(vox128, (H, H, H)) > H / 128 * 2).astype(np.float32)
        # vox32 = np.array(vox.revoxelized((H, H, H)).matrix)

        vox128 = self.vox_to_pseudosdf(vox128)
        vox32 = self.vox_to_pseudosdf(vox32)
        sample = {}
        sample['vox128'] = vox128[None]
        sample['nSdf'] = vox32[None]
        sample['hA'] = np.zeros([45,])
        sample['index'] = self.index_list[index]
        return sample

    def vox_to_pseudosdf(self, vox):
        vox = vox.astype(np.float32) - 0.5
        return -vox * 2


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
        print(batch['nSdf'].shape)
        H = 32
        sdf = -batch['nSdf'].reshape(bs, 1, H, H, H)
        vox = mesh_utils.cubify(sdf, th=0)
        mesh_utils.dump_meshes(osp.join(save_dir, '%d_vox' % i), vox)

        if i > 5:
            break