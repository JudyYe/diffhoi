"""follow ../vhoi/vos.sh that extract video annotations"""
from glob import glob
from random import sample
import shutil
import imageio
import numpy as np
import os
import os.path as osp
import pickle
from PIL import Image

import torch
from torchvision.transforms import ToTensor, transforms
from torch.utils.data import DataLoader, Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras

from jutils import image_utils, mesh_utils, geom_utils
from utils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o


# data_dir = '/glusterfs/yufeiy2/vhoi/100doh_detectron/by_obj/'
data_dir = '/glusterfs/yufeiy2/transfer/hoi_vid/100doh_detectron/by_obj/'
save_dir = '/glusterfs/yufeiy2/vhoi/syn_data/'
vis_dir = '/glusterfs/yufeiy2/vhoi/syn_data/vis'
device = 'cuda:0'

time_len = 30
H = W = 224



class Custom(Dataset):
    def __init__(self, seqname: str, data_dir='../data/'):
        super().__init__()
        self.data_dir = data_dir
        self.hand_mask_file = osp.join(self.data_dir, 'VidAnnotations', seqname, 'ppl.npy')
        self.mask_dir = sorted(glob(osp.join(self.data_dir, 'VidAnnotations', seqname, '*.png')))
        self.meta_dir = sorted(glob(os.path.join(self.data_dir, 'mocap_seq', seqname, 'mocap/*prediction_result.pkl')))
        self.image_dir = sorted(glob(osp.join(self.data_dir, 'JPEGImages', seqname, '*.jpg')))
        self.hand_meta = osp.join(self.data_dir, 'VidAnnotations', seqname, 'hand_inds_side.pkl')

        self.transform = transforms.ToTensor()
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

    def __len__(self):
        return len(self.mask_dir)

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
        for i in range(len(self)):
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
        print(self.anno['hand_mask'].shape, self.anno['hand_mask'].dtype)
        print(self.anno['obj_mask'][0].shape, self.anno['obj_mask'][0].dtype)
            
    def get_bbox(self, idx):
        return self.anno['bbox'][idx]
    
    def get_obj_mask(self, idx, bbox, key='obj_mask'):
        obj_mask = self.anno[key][idx]
        obj_mask = image_utils.crop_resize(obj_mask, bbox,return_np=False)
        return (self.transform(obj_mask) > 0).float()

    def get_image(self, idx, bbox):
        image = np.array(self.anno['image'][idx])
        image = image_utils.crop_resize(image, bbox, return_np=False)
        return self.transform(image) * 2 - 1

    def get_cam_fp(self, idx, bbox):
        pred_cam, hand_bbox_tl, bbox_scale = self.anno['cam'][idx]
        new_center = (bbox[0:2] + bbox[2:4]) / 2
        new_size = max(bbox[2:4] - bbox[0:2])
        cam, topleft, scale = image_utils.crop_weak_cam(
            pred_cam, hand_bbox_tl, bbox_scale, new_center, new_size)
        s, tx, ty = cam
        
        fx = 10
        f = torch.FloatTensor([fx, fx])
        p = torch.FloatTensor([0, 0])

        translate = torch.FloatTensor([tx, ty, fx/s])
        print('cam', cam, pred_cam, translate)
        return translate, f, p

    def get_cTh(self, idx, translate, ):
        hA = self.anno['hA'][idx]
        rot = self.anno['rot'][idx]
        _, joints = self.hand_wrapper(
            geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot[None])), 
            hA[None])
        
        cTh = geom_utils.axis_angle_t_to_matrix(
            rot, translate - joints[0, 5])
        return cTh

    def __getitem__(self, idx: int):
        sample = {}
        # add crop?? 
        
        sample['bbox'] = self.get_bbox(idx)  # target bbox
        translate, sample['cam_f'], sample['cam_p'] = self.get_cam_fp(idx, sample['bbox'])
        cTh = self.get_cTh(idx, translate)
        sample['cTh'] = geom_utils.matrix_to_se3(cTh)
        sample['image'] = self.get_image(idx, sample['bbox'])
        sample['obj_mask'] = self.get_obj_mask(idx, sample['bbox'])
        sample['hand_mask'] = self.get_obj_mask(idx, sample['bbox'], 'hand_mask')
        sample['hA'] = self.anno['hA'][idx]

        sample['rot'] = self.anno['rot'][idx]
        sample['index'] = self.anno['index'][idx]
        return sample   


def render(seqname, ):
    dataset = Custom(seqname, data_dir)
    dataloader = DataLoader(dataset, 1)
    hand_wrapper = ManopthWrapper().to(device)
    for data in dataloader:
        for k, v in data.items():
            try:
                data[k] = v.to(device)
            except AttributeError:
                pass

        cameras = PerspectiveCameras(data['cam_f'], data['cam_p'], device=device)
        cHand, _ = hand_wrapper(data['cTh'], data['hA'])

        iHand = mesh_utils.render_mesh(cHand, cameras)
        print(data['hand_mask'].shape, data['obj_mask'].shape)
        # mask = torch.cat([data['hand_mask'], data['obj_mask'], torch.ones_like(data['obj_mask'])], dim=1)
        mask = torch.cat([data['hand_mask'], data['hand_mask'], data['hand_mask']], dim=1)
        image_utils.save_images(mask, osp.join(vis_dir, seqname, 'inp'), mask=mask, bg=data['image'])
        image_utils.save_images(iHand['image'], osp.join(vis_dir, seqname, 'render'), mask=iHand['mask'], bg=data['image'])
        
        break
    
    # obj_mask, hand_mask, image, mask/ hands.npz, cameras_hoi.npz
    
    # RGB, masks
    


if __name__ == '__main__':
    render('study_v_im0FA2X6fp0_frame000043_0')
