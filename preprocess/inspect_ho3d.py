from inspect import FrameInfo
import imageio
import cv2
import numpy as np
import os
import os.path as osp
import pickle
from PIL import Image

import argparse
from pyparsing import FollowedBy
from sklearn.exceptions import DataDimensionalityWarning
import torch
from torchvision.transforms import ToTensor
import pytorch3d
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.structures import Pointclouds
from torchmetrics import ScaleInvariantSignalNoiseRatio
from models.frameworks.volsdf_hoi import MeshRenderer

from tools import render_view
from jutils import image_utils, mesh_utils, geom_utils
from utils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o
from utils.rend_util import load_K_Rt_from_P


# get psuedo mask
# cameras 

data_dir = '/checkpoint/yufeiy2/datasets/HO3D'
save_dir = '/checkpoint/yufeiy2/vhoi_out/syn_data/'
device = 'cuda:0'

time_len = 30
H = W = 224


def render(vid_index, start, dt=10, split='train'):
    hand_wrapper = ManopthWrapper().to(device)
    cnt = 0
    with open(osp.join(data_dir, '%s.txt' % split)) as fp:
        frame_dict = [line.strip() for line in fp if line.split('/')[0] == vid_index]
    frame_dict = set(frame_dict)
    t = start - 1
    
    image_list = []
    camera_dict = {'cTw': [], 'wTh': [], 'K_pix': [], 'hTo': [], 'onTo': []}
    hand_dict = {'hA': []}
    while t < len(frame_dict):
        if cnt >= time_len:
            break
        t += 1
        frame_index = '%04d' % t
        index = osp.join(vid_index, frame_index)
        if index not in frame_dict:
            print(index)
            continue
        t += dt - 1
        cnt += 1
        bg = Image.open(osp.join('../data/ho3d/%s/%s/rgb/%s.jpg' % (split, vid_index, frame_index)))
        # image = ToTensor()(image)[None]

        with open(osp.join('../data/ho3d/%s/%s/meta_plus/%s.pkl' % (split, vid_index, frame_index)), 'rb') as fp:
            anno = pickle.load(fp)
        hTo, cTh, hA, hHand, bbox_sq, f, p = get_crop(anno, hand_wrapper)

        # TODO: fix wTh across frame!!
        _, wTh = mesh_utils.center_norm_geom(hHand, 0, max_norm=2)
        wTh = wTh.get_matrix().transpose(-1, -2)
        hTw =  geom_utils.inverse_rt(mat=wTh, return_mat=True)
        cTw = cTh @ hTw

        mesh_file = osp.join('/checkpoint/yufeiy2/datasets/YCBObjects/models', anno['objName'], 'textured_simple.obj')
        oMesh = mesh_utils.load_mesh(mesh_file)
        print('oMesh', oMesh.verts_packed().max(0)[0] - oMesh.verts_packed().min(0)[0])
        _,  onTo = mesh_utils.center_norm_geom(oMesh, 0)  # obj norm
        onTo = onTo.get_matrix().transpose(-1, -2)
        
        hObj = mesh_utils.apply_transform(oMesh, hTo)
        hHoi = mesh_utils.join_scene([hObj, hHand])

        # render mask and image
        wHoi = mesh_utils.apply_transform(hHoi, wTh)
        cHoi = mesh_utils.apply_transform(wHoi, cTw)
        cameras = PerspectiveCameras(f, p).to(device)
        
        intr = mesh_utils.intr_from_ndc_to_screen(mesh_utils.get_k_from_fp(f, p), H, W)
        image = mesh_utils.render_mesh(cHoi.to(device), cameras, out_size=H)
        # image_utils.save_images(image['image'], osp.join(save_dir, 'tmp', '%s_%05d' % (vid_index, t)))

        bg = image_utils.crop_resize(np.array(bg), bbox_sq[0], H)
        bg = ToTensor()(bg)[None]
        img_np = image_utils.save_images(bg, osp.join(save_dir, '%s_%04d/image/' % (vid_index, start),  '%05d' % (cnt - 1)))
        image_utils.save_images(image['mask'], osp.join(save_dir, '%s_%04d/mask' % (vid_index, start),  '%05d' % (cnt - 1)))

        image_list.append(img_np)

        # render obj and hand mask
        hHoi = mesh_utils.join_scene_w_labels([hHand, hObj], 3)
        cHoi = mesh_utils.apply_transform(hHoi, cTw @ wTh)
        image = mesh_utils.render_mesh(cHoi.to(device), cameras, out_size=H)

        hand_mask = ((image['image'] * image['mask'])[:, 0:1] > 0.5).float()
        obj_mask = ((image['image'] * image['mask'])[:, 1:2] > 0.5).float()
        image_utils.save_images(hand_mask, osp.join(save_dir, '%s_%04d/hand_mask' % (vid_index, start),  '%05d' % (cnt - 1)))
        image_utils.save_images(obj_mask, osp.join(save_dir, '%s_%04d/obj_mask' % (vid_index, start),  '%05d' % (cnt - 1)))

        camera_dict['cTw'].append(cTw.cpu().detach().numpy()[0])
        camera_dict['wTh'].append(wTh.cpu().detach().numpy()[0])
        camera_dict['K_pix'].append(intr.cpu().detach().numpy()[0])
        camera_dict['hTo'].append(hTo.cpu().detach().numpy()[0])
        camera_dict['onTo'].append(onTo.cpu().detach().numpy()[0])
        hand_dict['hA'].append(hA.cpu().detach().numpy()[0])

    imageio.mimsave(osp.join(save_dir, '%s_%04d' % (vid_index, start), 'image.gif'), image_list)
    print('save to', osp.join(save_dir, '%s_%04d' % (vid_index, start), 'image.gif'))

    for k, v in camera_dict.items():
        camera_dict[k] = np.array(v)
        print(k, camera_dict[k].shape)
    for k, v in hand_dict.items():
        hand_dict[k] = np.array(v)
        print(hand_dict[k].shape)
    np.savez_compressed(osp.join(save_dir, '%s_%04d' % (vid_index, start), 'cameras_hoi.npz'), **camera_dict)
    np.savez_compressed(osp.join(save_dir, '%s_%04d' % (vid_index, start), 'hands.npz'), **hand_dict)


def get_crop(anno, hand_wrapper):
    pose = torch.FloatTensor(anno['handPose'][None])  # handTrans
    trans = torch.FloatTensor(anno['handTrans'][None])


    hA = pose[..., 3:]
    rot = pose[..., :3]
    rot, trans = cvt_axisang_t_i2o(rot, trans)
    wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)

    wTo = geom_utils.axis_angle_t_to_matrix(
        torch.FloatTensor([anno['objRot'].reshape(3)]), 
        torch.FloatTensor([anno['objTrans'].reshape(3)]))
    hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo

    rot = torch.FloatTensor([[[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]])
    cTw = geom_utils.rt_to_homo(rot, )
    cTh = cTw @ wTh

    cHand, cJoints = hand_wrapper(geom_utils.matrix_to_se3(cTw @ wTh).to(device), hA.to(device))
    hHand, _ = hand_wrapper(None, hA.to(device))
    cam_intr = torch.FloatTensor(anno['camMat'][None])

    cCorner = mesh_utils.apply_transform(torch.FloatTensor(anno['objCorners3D'][None]), cTw)
    bboxj2d = minmax(torch.cat([proj3d(cJoints.cpu(), cam_intr), proj3d(cCorner, cam_intr)], dim=1))
    bbox_sq = square_bbox(bboxj2d, pad=0.1)
    
    cam_intr = crop_cam(cam_intr, bbox_sq, 1)
    f, p = get_K(cam_intr, 1, 1)
    return hTo, cTh, hA, hHand.cpu(), bbox_sq, f, p


def get_K(cam, H, W):
    """
    Args:
        cam ([type]): [description]
        H ([type]): [description]
        W ([type]): [description]
    Returns:
        [type]: (N, 2) (N, 2)
    """
    k = torch.FloatTensor([[
        [2 / H, 0, -1],
        [0, 2 / W, -1],
        [0, 0,      1],
    ]])
    out = k @ cam
    f = torch.diagonal(out, dim1=-1, dim2=-2)[..., :2]
    p = out[..., 0:2, 2]
    
    return f, p

def crop_cam(cam_intr, bbox_sq, H):
    x1y1 = bbox_sq[..., 0:2]
    dxy = bbox_sq[..., 2:] - bbox_sq[..., 0:2]
    t_mat = torch.FloatTensor([[
        [1, 0, -x1y1[0, 0]],
        [0, 1, -x1y1[0, 1]],
        [0, 0, 1],
    ]]).to(cam_intr)
    s_mat = torch.FloatTensor([[
        [H / dxy[0, 0], 0, 0],
        [0, H / dxy[0, 1], 0],
        [0, 0, 1],
    ]]).to(cam_intr)
    mat = s_mat @ t_mat @ cam_intr
    return mat


def minmax(pts2d):
    x1y1 = torch.min(pts2d, dim=-2)[0]  # (..., P, 2)
    x2y2 = torch.max(pts2d, dim=-2)[0]  # (..., P, 2)
    return torch.cat([x1y1, x2y2], dim=-1)  # (..., 4)

def square_bbox(bbox, pad=0):
    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    center = (x1y1 + x2y2) / 2 
    half_w = torch.max((x2y2 - x1y1) / 2, dim=-1)[0]
    half_w = half_w * (1 + 2 * pad)
    bbox = torch.cat([center - half_w, center + half_w], dim=-1)
    return bbox


def proj3d(points, cam):
    p2d = points.cpu() @ cam.cpu().transpose(-1, -2)
    p2d = p2d[..., :2] / p2d[..., 2:3]
    return p2d


if __name__ == '__main__':
    # render('MDF10', 0, 10)
    # render('SMu1', 650, 10)
    
    render('SMu41', 0, 10)
    # sugar box
    render('SS2', 0, 10)
    # 006_mustard_bottle
    render('SM2', 0, 10)
    # scissor
    render('GSF11', 0, 10)
    render('GSF11', 1000, 10)
    # 003_cracker_box
    render('MC2', 0, 10)
    # banana
    render('BB12', 0, 10)

    # render('AP12', 50, 10, split='evaluation')
    # AP12/0051
    # flow()