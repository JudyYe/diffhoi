import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

from .DTU import SceneDataset
from utils.io_util import load_flow, load_mask, load_rgb, glob_imgs
from utils.rend_util import rot_to_quat, load_K_Rt_from_P
from jutils import mesh_utils


class HoiDataset(SceneDataset):
    # NOTE: jianfei: modified from IDR.   https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 cam_file=None,
                 scale_radius=-1):
        super().__init__(train_cameras, data_dir, downscale, cam_file, scale_radius)
        self.hand = mesh_utils.load_mesh(data_dir, 'hand.obj')

    def __len__(self):
        return self.n_images - 1

    def __getitem__(self, idx):
        idx, sample, ground_truth = super().__getitem__(idx)
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)
        sample['hand'] = self.hand

if __name__ == "__main__":
    dataset = SceneDataset(False, './data/DTU/scan40')
    c2w = dataset.get_gt_pose(scaled=True).data.cpu().numpy()
    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()
    print('intrinsic??', camera_matrix, camera_matrix.shape)
    print('extrinsic', extrinsics, extrinsics.shape)
    print(dataset.H, dataset.W)
    # print(next(iter(dataset))[-1]['rgb'].size())
    from tools.vis_camera import visualize
    visualize(camera_matrix, extrinsics)

    cam_loc = c2w[..., :3, 3]
    dist = np.sqrt((cam_loc**2).sum(-1))
    print(dist)