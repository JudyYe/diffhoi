from __future__ import print_function

import os
import os.path as osp
import pickle
import torch
import numpy as np
import tqdm
from jutils import mesh_utils, geom_utils

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, Rotate, Translate


def solve_rt(src_mesh, dst_mesh):
    """
    (N, P, 3), (N, P, 3)
    """
    device = src_mesh.device
    src_centroid = torch.mean(src_mesh, -2, keepdim=True)
    dst_centroid = torch.mean(dst_mesh, -2, keepdim=True)
    src_bar = (src_mesh - src_centroid)
    dst_bar = (dst_mesh - dst_centroid)
    cov = torch.bmm(src_bar.transpose(-1, -2), dst_bar)
    u, s, v = torch.svd(cov)
    vh = v.transpose(-1, -2)
    rot_t = torch.matmul(u, vh)
    rot = rot_t.transpose(-1, -2)  # v, uh

    trans = dst_centroid - torch.matmul(src_centroid, rot_t)  # (N, 1, 3)?

    rot = Rotate(R=rot_t, device=device)
    trans = Translate(trans.squeeze(1), device=device)

    rt = rot.compose(trans)

    return rt


def extract_rt_pose(meta_info, pose_wrapper):
    device = 'cpu'
    cam_extr = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0, 0, 0, 1]
        ]
    ).astype(np.float32)
    cTw = Transform3d(device=device, matrix=torch.FloatTensor([cam_extr]).transpose(1, 2).to(device))
    wTo = Transform3d(device=device, matrix=torch.FloatTensor([meta_info['affine_transform']]).transpose(1, 2).to(device))
    cTo = wTo.compose(cTw)
    cTo_mat = cTo.get_matrix().transpose(-1, -2).cpu()

    # src mesh
    zeros = torch.zeros([1, 3], device=device, dtype=torch.float32)
    art_pose = torch.FloatTensor([meta_info['pca_pose']]).to(device)
    art_pose = pose_wrapper.pca_to_pose(art_pose) - pose_wrapper.hand_mean
    hHand, _ = pose_wrapper(None, art_pose, zeros, zeros, mode='inner')
    # dst mesh
    wVerts = torch.FloatTensor([meta_info['verts_3d']]).to(device)
    wHand = Meshes(wVerts, pose_wrapper.hand_faces)

    wTh = solve_rt(hHand.verts_padded(), wHand.verts_padded())  
    cTh = wTh.compose(cTw)
    cTh_mat = cTh.get_matrix().transpose(-1, -2).cpu()

    return cTh_mat, art_pose.cpu(), cTo_mat


class Obman:
    def __init__(self, split='val', dataset='obman', data_dir='/glusterfs/yufeiy2/fair/data/'):
        super().__init__()
        self.split = split
        self.dataset = dataset
        self.data_dir = osp.join(data_dir, 'obman')
        self.cache = True
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],
            'hTo': [],
        }
        self.obj2mesh = {}

        self.cache_file = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s.pkl' % (dataset, self.split))
        self.cache_mesh = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s_mesh.pkl' % (dataset, self.split))

        self.shape_dir = os.path.join('/glusterfs/yufeiy2/download_data/ShapeNetCore.v2/', '{}', '{}', 'models', 'model_normalized.obj')
        self.meta_dir = os.path.join(self.data_dir, split, 'meta', '{}.pkl')
    
    def preload_anno(self, load_keys=[]):
        for key in load_keys:
            self.anno[key] = []
        
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!', self.cache_file)
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:
            print('making cache in %s' % self.cache_file)
            index_list = [line.strip() for line in open(osp.join(self.data_dir, '%s.txt' % self.split))]
            for i, index in enumerate(tqdm.tqdm(index_list)):
                dset = self.dataset
                if 'mini' in dset and i >= int(dset.split('mini')[-1]):
                    break

                meta_path = self.meta_dir.format(index)
                with open(meta_path, "rb") as meta_f:
                    meta_info = pickle.load(meta_f)

                global_rt, art_pose, obj_pose = extract_rt_pose(meta_info, self.pose_wrapper)

                self.anno['index'].append(index)
                self.anno['cad_index'].append(osp.join(meta_info["class_id"], meta_info["sample_id"]))
                cTo = obj_pose
                cTh = global_rt
                hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True)
                hTo = torch.matmul(hTc, cTo)

                self.anno['hTo'].append(hTo[0])
                self.anno['hA'].append(art_pose.detach().numpy()[0])

            os.makedirs(osp.dirname(self.cache_file), exist_ok=True)
            print('save cache')
            pickle.dump(self.anno, open(self.cache_file, 'wb'))

    def preload_mesh(self):
        if self.cache and osp.exists(self.cache_mesh):
            print('!! Load from cache !!', self.cache_mesh)
            self.obj2mesh = pickle.load(open(self.cache_mesh, 'rb'))
        else:
            self.obj2mesh = {}
            print('load mesh', self.cache_mesh)
            for i, cls_id in tqdm.tqdm(enumerate(self.anno['cad_index']), total=len(self.anno['cad_index'])):
                key = cls_id
                cls, id = key.split('/')
                if key not in self.obj2mesh:
                    fname = self.shape_dir.format(cls, id)
                    self.obj2mesh[key] = mesh_utils.load_mesh(fname, scale_verts=1)
            print('save cache')
            pickle.dump(self.obj2mesh, open(self.cache_mesh, 'wb'))


def get_anno_split(split):
    data_dir = '/glusterfs/yufeiy2/fair/data/'
    obman = Obman(split, )
    obman.preload_anno()
    obman.preload_mesh()
    anno = obman.anno
    anno['sdf_file'] = []
    for i in range(len(anno['index'])):
        sdf_dir = osp.join(data_dir, 'mesh_sdf/SdfSamples/', 'obman', 'all')
        cad_idx = anno['cad_index'][i]
        filename = osp.join(sdf_dir, cad_idx + '.npz')
        assert osp.exists(filename), 'Not exists %s' % filename

        anno['sdf_file'].append(filename)
    return anno, obman.obj2mesh