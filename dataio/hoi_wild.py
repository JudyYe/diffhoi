import os
import os.path as osp
from turtle import down
import imageio
import torch
import numpy as np
from glob import glob
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from utils.hand_utils import ManopthWrapper, get_nTh

from jutils import geom_utils, image_utils, mesh_utils

class SceneDataset(Dataset):
    def __init__(self, 
        train_cameras,
        data_dir,
        downscale=1.,   # [H, W]
        cam_file=None,
        scale_radius=-1, args=dict()):

        # seqname: str, data_dir='../data/'):
        super().__init__()
        self.args = args
        self.data_dir, seqname = data_dir
        self.train_cameras = train_cameras

        self.hand_mask_file = osp.join(self.data_dir, 'VidAnnotations', seqname, 'ppl.npy')
        self.mask_dir = sorted(glob(osp.join(self.data_dir, 'VidAnnotations', seqname, '*.png')))
        self.meta_dir = sorted(glob(os.path.join(self.data_dir, 'mocap_seq', seqname, 'mocap/*prediction_result.pkl')))
        self.image_dir = sorted(glob(osp.join(self.data_dir, 'JPEGImages', seqname, '*.jpg')))
        self.hand_meta = osp.join(self.data_dir, 'VidAnnotations', seqname, 'hand_inds_side.pkl')

        self.H = self.W = 224 // downscale
        tmp_rgb = imageio.imread(self.image_dir[0])
        self.orig_size = max(tmp_rgb.shape[0], tmp_rgb.shape[1])
        self.downscale = downscale
        self.max_cam_norm = -1

        self.hand_wrapper = ManopthWrapper().to('cpu')
        self.hand_mean = self.hand_wrapper.hand_mean
        assert len(self.mask_dir) == len(self.meta_dir), '%d %d %d' % (len(self.mask_dir), len(self.meta_dir), len(self.image_dir))
        assert len(self.mask_dir) == len(self.image_dir)
        self.anno = {
            'index': [],  # per grasp
            'image': [],
            'obj_mask': [],
            'hand_mask': [],
            
            'hA': [],
            'rot': [],
            'bbox': [],
            'cam': [],
        }    
        self.preload_anno()
        self.preprocess()

    def __len__(self):
        return len(self.mask_dir) - 1

    def preprocess(self):  
        self.sample_list = []
        for i in range(len(self) + 1):
            sample = self._get_idx(i)
            self.sample_list.append(sample)
            self.max_cam_norm = max(sample['translate'][..., -1].item(), self.max_cam_norm)

        self.flow_fw = torch.zeros([len(self), self.H * self.W, 2])
        self.flow_bw = []

    def preload_anno(self):
        if not osp.exists(self.hand_meta):
            print(self.hand_meta)
            ppl = 0
        else:
            hand_meta = pickle.load(open(self.hand_meta, 'rb'))
            ppl = hand_meta['hand_inds']
            if len(ppl) == 0:
                ppl = 0
            else:
                ppl = ppl[0]
        self.anno['hand_mask'] = np.load(self.hand_mask_file)[ppl].astype(np.uint8) * 255  # [T, H, W]?
        for i in range(len(self.mask_dir)):
            self.anno['index'].append('%05d' % i)

            image = Image.open(self.image_dir[i])
            self.anno['image'].append(image)

            mask = np.array(Image.open(self.mask_dir[i]))
            if mask.ndim > 2:
                mask = [..., 0]
            obj_mask = (mask > 0)
            mask = (obj_mask).astype(np.uint8) * 255
            self.anno['obj_mask'].append(mask)

            # hand pose
            meta_path = self.meta_dir[i]
            with open(meta_path, "rb") as meta_f:
                anno = pickle.load(meta_f)['pred_output_list']
            one_hand = anno[ppl]['right_hand']

            pose = torch.FloatTensor(one_hand['pred_hand_pose'])
            rot, hA = pose[..., :3], pose[..., 3:]
            hA = hA + self.hand_mean

            obj_bbox = image_utils.mask_to_bbox(mask, 'med')
            x1, y1 = one_hand['bbox_top_left'] 
            bbox_len = 224 / one_hand['bbox_scale_ratio']
            x2, y2 = x1 + bbox_len, y1 + bbox_len
            
            hand_bbox = np.array([x1,y1, x2, y2])
            hoi_bbox = image_utils.joint_bbox(obj_bbox, hand_bbox)
            hoi_bbox = image_utils.square_bbox(hoi_bbox, 0.2)
            
            self.anno['bbox'].append(hoi_bbox)
            self.anno['rot'].append(rot[0])
            self.anno['cam'].append([one_hand['pred_camera'], one_hand['bbox_top_left'], one_hand['bbox_scale_ratio']])
            self.anno['hA'].append(hA[0])
            
    def get_bbox(self, idx):
        return self.anno['bbox'][idx]
    
    def get_obj_mask(self, idx, bbox, key='obj_mask'):
        obj_mask = self.anno[key][idx]
        obj_mask = image_utils.crop_resize(obj_mask, bbox,return_np=True, final_size=self.H)
        return torch.FloatTensor((obj_mask > 0).reshape(-1))

    def get_image(self, idx, bbox):
        image = np.array(self.anno['image'][idx])
        image = image_utils.crop_resize(image, bbox, return_np=True, final_size=self.H)
        return torch.FloatTensor(image.reshape(-1, 3).astype(np.float32) / 255)

    def get_cam_fp(self, idx, bbox):
        pred_cam, hand_bbox_tl, bbox_scale = self.anno['cam'][idx]
        new_center = (bbox[0:2] + bbox[2:4]) / 2
        new_size = max(bbox[2:4] - bbox[0:2])
        cam, topleft, scale = image_utils.crop_weak_cam(
            pred_cam, hand_bbox_tl, bbox_scale, new_center, new_size)
        s, tx, ty = cam
        
        # fx = 10
        fx = self.args.data.focal * self.orig_size / new_size
        f = torch.FloatTensor([fx, fx])
        p = torch.FloatTensor([0, 0])

        translate = torch.FloatTensor([tx, ty, fx/s])
        return translate, f, p

    def get_cTh(self, idx, translate, ):
        hA = self.anno['hA'][idx]
        rot = self.anno['rot'][idx]
        _, joints = self.hand_wrapper(
            geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot[None])), 
            hA[None])
        
        cTh = geom_utils.axis_angle_t_to_matrix(
            rot, translate - joints[0, 5])
        nTh = get_nTh(center=joints[:,5])[0]
        return cTh, nTh

    def _get_idx(self, idx: int):
        sample = {}
        
        sample['bbox'] = self.get_bbox(idx)  # target bbox
        translate, sample['cam_f'], sample['cam_p'] = self.get_cam_fp(idx, sample['bbox'])
        cTh, nTh = self.get_cTh(idx, translate)
        sample['cTh'] = cTh
        sample['nTh'] = nTh
        sample['image'] = self.get_image(idx, sample['bbox'])
        sample['obj_mask'] = self.get_obj_mask(idx, sample['bbox'])
        sample['hand_mask'] = self.get_obj_mask(idx, sample['bbox'], 'hand_mask')
        sample['hA'] = self.anno['hA'][idx]
        sample['translate'] = translate

        sample['rot'] = self.anno['rot'][idx]
        return sample
    
    def get_K_pix(self, cam_f, cam_p):

        K = mesh_utils.get_k_from_fp(cam_f[None], cam_p[None])
        K_pix = mesh_utils.intr_from_ndc_to_screen(K, self.H, self.W)[0]
        return K_pix

    def __getitem__(self, idx):
        idx_n = idx + 1
        out = self.sample_list[idx]
        out_n = self.sample_list[idx_n]

        sample = {
            "obj_mask": out['obj_mask'],
            "hand_mask": out['hand_mask'],
            "object_mask": (out['obj_mask'] + out['hand_mask']).clamp(max=1).reshape(-1),
            
            "hA": out['hA'],
            "hA_n": out_n['hA'],

            "onTo": out['nTh'],
            "onTo_n": out_n['nTh'],

            "c2w": geom_utils.inverse_rt(mat=out['cTh'], return_mat=True),
            "c2w_n": geom_utils.inverse_rt(mat=out_n['cTh'], return_mat=True),

            "intrinsics": self.get_K_pix(out['cam_f'], out['cam_p']),
            "intrinsics_n": self.get_K_pix(out_n['cam_f'], out_n['cam_p']),

            'flow_fw': self.flow_fw[idx]
        }

        sample["inds_n"] = idx_n        
        sample["hTo"] = sample['hTo_n'] = sample['wTh'] = sample['wTh_n'] = torch.eye(4)

        ground_truth = {}
        ground_truth['rgb'] = out['image']
        ground_truth['hA'] = out['hA']
        return idx, sample, ground_truth