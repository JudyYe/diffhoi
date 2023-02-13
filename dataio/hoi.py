import logging
import os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

from utils.io_util import load_flow, load_mask, load_rgb, glob_imgs
from jutils import geom_utils, image_utils, mesh_utils


class SceneDataset(torch.utils.data.Dataset):
    # NOTE: for HO3D
    # jianfei: modified from IDR.   https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 cam_file=None,
                 scale_radius=-1, args=dict()):

        assert os.path.exists(data_dir), "Data directory is empty %s" % data_dir

        if osp.exists(osp.join(data_dir, 'oObj.obj')):
            self.oObj = mesh_utils.load_mesh(osp.join(data_dir, 'oObj.obj'))
        else:
            self.oObj = None
        self.instance_dir = data_dir
        self.train_cameras = train_cameras

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(glob_imgs(mask_dir))
        fw_dir = '{0}/FlowFW'.format(self.instance_dir)
        fw_paths = sorted(glob(os.path.join(fw_dir, '*.npz')))
        bw_dir = '{0}/FlowBW'.format(self.instance_dir)
        bw_paths = sorted(glob(os.path.join(bw_dir, '*.npz')))

        self.n_images = min(len(image_paths), args.data.setdefault('len', 10000))
        
        # determine width, height
        self.downscale = downscale
        tmp_rgb = load_rgb(image_paths[0], downscale)
        _, self.H, self.W = tmp_rgb.shape

        # load camera and pose
        self.cam_file = '{0}/cameras_hoi.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        self.wTc = geom_utils.inverse_rt(mat=torch.from_numpy(camera_dict['cTw']).float(), return_mat=True)
        N = len(self.wTc)
        idty = torch.eye(4)[None].repeat(N, 1, 1)
        if 'wTh' in camera_dict:
            self.wTh = torch.from_numpy(camera_dict['wTh']).float()  # a scaling mat that recenter hoi to -1, 1? 
        else:
            self.wTh = idty
        self.hTo = camera_dict.get('hTo', idty)
        scale = idty.clone()
        scale[:, 0, 0] = scale[:, 1, 1] = scale[:, 2, 2] = 10
        self.onTo = camera_dict.get('onTo', scale)
        self.intrinsics_all = torch.from_numpy(camera_dict['K_pix']).float()  # (N, 4, 4)
        # downscale * self.H is the orignal height before resize. 
        self.intrinsics_all = mesh_utils.intr_from_screen_to_ndc(
            self.intrinsics_all, downscale* self.H, downscale * self.W)

        # downscale intrinsics
        # self.intrinsics_all[..., 0, 2] /= downscale
        # self.intrinsics_all[..., 1, 2] /= downscale
        # self.intrinsics_all[..., 0, 0] /= downscale
        # self.intrinsics_all[..., 1, 1] /= downscale

        self.scale_cam = 1
        # calculate cam distance
        cam_center_norms = []
        for wTc in self.wTc:
            cam_center_norms.append(np.linalg.norm(wTc[:3,3].detach().numpy()))
        max_cam_norm = max(cam_center_norms)
        self.max_cam_norm = max_cam_norm  # in camera metric??? 
        print('max cam norm', self.max_cam_norm)

        # TODO: crop??!!!
        self.rgb_images = []
        for path in tqdm(image_paths, desc='loading images...'):
            rgb = load_rgb(path, downscale)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for path in mask_paths:
            object_mask = load_mask(path, downscale)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        self.flow_fw = []
        for path in tqdm(fw_paths):
            if osp.exists(path):
                flow = load_flow(path, downscale)
                flow = flow.reshape(-1, 2)
            self.flow_fw.append(torch.from_numpy(flow).float())
        if len(self.flow_fw) == 0:
            print('No flow!')
            self.flow_fw = torch.zeros([self.n_images - 1, self.H * self.W, 2])
        self.flow_bw = []

        # load hand 
        hands = np.load('{0}/hands.npz'.format(self.instance_dir))
        self.hA = torch.from_numpy(hands['hA']).float().squeeze(1)
        noise = args.data.noise
        if args.hA.mode == 'learn':
            logging.info('jitter hand a bit!')
            self.hA_inp = self.hA + (torch.rand([45]) * noise * 2 - noise)
        else:
            self.hA_inp = self.hA

        # self.hand = mesh_utils.load_mesh(osp.join(data_dir, 'hand.obj'), scale_verts=self.scale_cam)
        # self.hand.textures = mesh_utils.pad_texture(self.hand, 'yellow')

        image_dir = '{0}/obj_mask'.format(self.instance_dir)
        obj_paths = sorted(glob_imgs(image_dir))
        image_dir = '{0}/hand_mask'.format(self.instance_dir)
        hand_paths = sorted(glob_imgs(image_dir))

        self.obj_masks = []
        for path in obj_paths:
            object_mask = load_mask(path, downscale)
            object_mask = object_mask.reshape(-1)
            self.obj_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        self.hand_masks = []
        self.hand_contours = []
        for path in hand_paths:
            object_mask = load_mask(path, downscale)
            hand_c = image_utils.sample_contour(object_mask > 0).astype(np.float32) / object_mask.shape[0] * 2 - 1
            self.hand_contours.append(hand_c)
            object_mask = object_mask.reshape(-1)
            self.hand_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        if len(self.object_masks) == 0:
            for obj_mask, hand_mask in zip(self.obj_masks, self.hand_masks):
                self.object_masks.append(torch.logical_or(obj_mask, hand_mask))

    def __len__(self):
        return self.n_images - 1

    def __getitem__(self, idx): 
        # TODO: support crop!
        idx_n = idx + 1
        sample = {
            "object_mask": self.object_masks[idx],
            "intrinsics":  self.intrinsics_all[idx],
            "intrinsics_n": self.intrinsics_all[idx_n],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        ground_truth["rgb"] = self.rgb_images[idx]
        sample["object_mask"] = self.object_masks[idx]

        sample['flow_fw'] = self.flow_fw[idx]
        sample['inds_n'] = idx_n
        
        if not self.train_cameras:
            sample["c2w"] = self.wTc[idx]
            sample['c2w_n'] = self.wTc[idx_n]

            sample['wTh'] = self.wTh[idx]
            sample['wTh_n'] = self.wTh[idx_n]

            sample['hTo'] = self.hTo[idx]
            sample['hTo_n'] = self.hTo[idx_n]
            
            sample['onTo'] = self.onTo[idx]
            sample['onTo_n'] = self.onTo[idx_n]

        sample['obj_mask'] = self.obj_masks[idx]
        sample['hand_mask'] = self.hand_masks[idx]
        sample['hand_contour'] = self.hand_contours[idx]
        sample['hA'] = self.hA_inp[idx]
        sample['hA_n'] = self.hA_inp[idx_n]

        ground_truth['hA'] = self.hA[idx]
        return idx, sample, ground_truth

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