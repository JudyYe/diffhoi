# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import json
import os
import os.path as osp
import pickle
import numpy as np
import tqdm
from PIL import Image
import pandas as pd
import torch
from pytorch3d.transforms.transform3d import Rotate, Scale, Translate

from jutils import mesh_utils, geom_utils
from jutils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o



class HO3D:
    def __init__(self, split='val', dataset='mow', data_dir='/glusterfs/yufeiy2/fair/data/'):
        super().__init__()
        self.split = split
        self.dataset = dataset
        self.data_dir = osp.join(data_dir, 'mow')
        self.cache = True
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
                'hA': [],
            'hTo': [],
        }
        
        self.shape_dir = os.path.join('/glusterfs/yufeiy2/download_data/YCBObjects//models', '{}', 'textured_simple.obj')
        meta_folder = 'meta_plus'
        self.use_gt = 'plus'

        self.cache_file = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s_%s.pkl' % (dataset, self.split, self.use_gt))
        self.cache_mesh = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s_mesh.pkl' % (dataset, self.split))
        self.meta_dir = os.path.join(self.data_dir, '{}', '{}', meta_folder, '{}.pkl')
        

    def preload_anno(self, load_keys=[]):
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!', self.cache_file)
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:
            print('creating cahce', self.meta_dir)
            # filter 1e-3
            df = pd.read_csv(osp.join(self.data_dir, '%s%s.csv' % (self.split, self.suf)))
            sub_df = df[df['dist'] < 5]
            sub_df = sub_df[sub_df['vid'] == 'MDF11']
            # sub_df = sub_df[sub_df['frame'] >= 350]
            
            print(len(df), '-->', len(sub_df))
            index_list = sub_df['index']
            folder_list = sub_df['split']

            for i, (index, folder) in enumerate(tqdm.tqdm(zip(index_list, folder_list))):
                # if (self.split == 'test' or self.split == 'val') and i % 10 != 0:
                    # continue
                index = (folder, index.split('/')[0], index.split('/')[1])
                meta_path = self.meta_dir.format(*index)
                with open(meta_path, "rb") as meta_f:
                    anno = pickle.load(meta_f)

                self.anno['index'].append(index)
                self.anno['cad_index'].append(anno["objName"])
                pose = torch.FloatTensor(anno['handPose'])[None]  # handTrans
                trans = torch.FloatTensor(anno['handTrans'].reshape(3))[None]
                hA = pose[..., 3:]
                rot = pose[..., :3]
                rot, trans = cvt_axisang_t_i2o(rot, trans)
                wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)

                wTo = geom_utils.axis_angle_t_to_matrix(
                    torch.FloatTensor([anno['objRot'].reshape(3)]), 
                    torch.FloatTensor([anno['objTrans'].reshape(3)]))
                hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo

                self.anno['hTo'].append(hTo[0])
                self.anno['hA'].append(hA[0])
                
            # os.makedirs(osp.dirname(self.cache_file), exist_ok=True)
            # print('save cache')
            # pickle.dump(self.anno, open(self.cache_file, 'wb'))


    def preload_mesh(self):
        if self.cache and osp.exists(self.cache_mesh):
            print('!! Load from cache !!')
            self.obj2mesh = pickle.load(open(self.cache_mesh, 'rb'))
        else:
            self.obj2mesh = {}
            print('load mesh')
            for i, cls_id in tqdm.tqdm(enumerate(self.anno['cad_index']), total=len(self.anno['cad_index'])):
                key = cls_id
                if key not in self.obj2mesh:
                    fname = self.shape_dir.format(cls_id)
                    self.obj2mesh[key] = mesh_utils.load_mesh(fname, scale_verts=1)
            print('save cache')
            pickle.dump(self.obj2mesh, open(self.cache_mesh, 'wb'))
