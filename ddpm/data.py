import logging
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset


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

        