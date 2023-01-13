from glob import glob
import imageio
import cv2
import numpy as np
import os
import os.path as osp
import pickle
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from pytorch3d.renderer.cameras import PerspectiveCameras

from jutils import image_utils, mesh_utils, geom_utils
from jutils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o, cvt_axisang_t_o2i


# get psuedo mask
# cameras 

data_dir = '/home/yufeiy2/scratch/data/HO3D/'
save_dir = '/home/yufeiy2/scratch/result/syn_data/'
shape_dir = '/home/yufeiy2/scratch/data/YCB/models'
device = 'cuda:0'

time_len = 30
H = W = 224

def render_from_original(split='train'):
    hand_wrapper = ManopthWrapper().to(device)
    cnt = 0
    with open(osp.join(data_dir, '%s.txt' % split)) as fp:
        frame_dict = [line.strip() for line in fp]
    
    for i, index in enumerate(frame_dict):
        vid_index = osp.dirname(index)
        frame_index = osp.basename(index)
        img_file = osp.join(f'{data_dir}/{split}/{vid_index}/rgb/{frame_index}.png')
        seg_file = osp.join(f'{data_dir}/{split}/{vid_index}/seg/{frame_index}.jpg')
        meta_file = osp.join(f'{data_dir}/{split}/{vid_index}/meta/{frame_index}.pkl')

        with open(meta_file, 'rb') as fp:
            anno = pickle.load(fp)
        
        bg = imageio.imread(img_file)
        seg = imageio.imread(seg_file)
        HH, WW = bg.shape[:2]
        print(seg.shape, bg.shape)
        seg = cv2.resize(seg, (WW, HH))
        cHand, bbox_sq = get_cHand(anno, hand_wrapper)
        cam_intr = torch.FloatTensor(anno['camMat'][None])
        cam_intr = image_utils.crop_cam_intr(cam_intr, bbox_sq, (H, W))
        cam_intr = mesh_utils.intr_from_screen_to_ndc(cam_intr, H, W)

        f, p = mesh_utils.get_fxfy_pxpy(cam_intr)
        cameras = PerspectiveCameras(f, p).to(device)
        image = mesh_utils.render_mesh(cHand, cameras, out_size=H)

        bg = image_utils.crop_resize(bg, bbox_sq[0], H)
        seg = image_utils.crop_resize(seg, bbox_sq[0], H)
        bg = ToTensor()(bg)[None]
        seg = ToTensor()(seg)[None]
        r = 0.7
        fg = bg.cpu() * image['mask'].cpu() + (r + (1-r) * bg) * (1-image['mask'].cpu())
        image_utils.save_images(fg, osp.join(save_dir, f'{vid_index}_{frame_index}_mask'))
        fg = seg.cpu() * image['mask'].cpu() + (r + (1-r) * seg) * (1-image['mask'].cpu())
        image_utils.save_images(fg, osp.join(save_dir, f'{vid_index}_{frame_index}_seg'))
        if i > 5:
            break


def render(vid_index, start, dt=10, split='train', skip=True):
    save_index = '%s_%04d_dt%02d' % (vid_index, start, dt)
    if osp.exists(osp.join(save_dir, save_index, 'hands.npz')) and skip:
        return
    hand_wrapper = ManopthWrapper().to(device)
    cnt = 0
    with open(osp.join(data_dir, '%s.txt' % split)) as fp:
        frame_dict = [line.strip() for line in fp if line.split('/')[0] == vid_index]
    frame_dict = set(frame_dict)
    t = start - 1
    
    image_list = []
    camera_dict = {'cTw': [], 'wTh': [], 'K_pix': [], 'hTo': [], 'onTo': []}
    hand_dict = {'hA': [], 'beta': []}
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
        bg = Image.open(osp.join(data_dir, '%s/%s/rgb/%s.png' % (split, vid_index, frame_index)))
        seg = Image.open(osp.join(data_dir, '%s/%s/seg/%s.jpg' % (split, vid_index, frame_index))).resize(bg.size)

        with open(osp.join(data_dir, '%s/%s/meta/%s.pkl' % (split, vid_index, frame_index)), 'rb') as fp:
            anno = pickle.load(fp)


        hTo, cTh, hA, hHand, bbox_sq, cam_intr_crop, beta = get_crop(anno, hand_wrapper)

        wTh = torch.eye(4)[None].to(device)
        cTw = cTh

        mesh_file = osp.join(shape_dir, anno['objName'], 'textured_simple.obj')
        oMesh = mesh_utils.load_mesh(mesh_file)
        _,  onTo = mesh_utils.center_norm_geom(oMesh, 0)  # obj norm
        onTo = onTo.get_matrix().transpose(-1, -2)
        
        hObj = mesh_utils.apply_transform(oMesh, hTo)
        hHoi = mesh_utils.join_scene([hObj, hHand])

        # render mask and image
        cHoi = mesh_utils.apply_transform(hHoi, cTh)
        f, p = mesh_utils.get_fxfy_pxpy(mesh_utils.intr_from_screen_to_ndc(cam_intr_crop, H, W))
        cameras = PerspectiveCameras(f, p).to(device)
        
        intr = cam_intr_crop

        image = mesh_utils.render_mesh(cHoi.to(device), cameras, out_size=H)

        bg = image_utils.crop_resize(np.array(bg), bbox_sq[0], H)
        seg = image_utils.crop_resize(np.array(seg), bbox_sq[0], H)
        bg = ToTensor()(bg)[None]
        seg = ToTensor()(seg)[None]
        img_np = image_utils.save_images(bg, osp.join(save_dir, '%s/image/' % (save_index),  '%05d' % (cnt - 1)))
        image_utils.save_images(image['mask'], osp.join(save_dir, '%s/mask' % (save_index),  '%05d' % (cnt - 1)))

        image_list.append(img_np)

        # # render obj and hand mask
        # hHoi = mesh_utils.join_scene_w_labels([hHand, hObj], 3)
        # cHoi = mesh_utils.apply_transform(hHoi, cTh)
        # image = mesh_utils.render_mesh(cHoi.to(device), cameras, out_size=H)

        # hand_mask = ((image['image'] * image['mask'])[:, 0:1] > 0.5).float()
        # obj_mask = ((image['image'] * image['mask'])[:, 1:2] > 0.5).float()
        # image_utils.save_images(hand_mask, osp.join(save_dir, '%s/hand_mask' % (save_index),  '%05d' % (cnt - 1)))
        # image_utils.save_images(obj_mask, osp.join(save_dir, '%s/obj_mask' % (save_index),  '%05d' % (cnt - 1)))

        hand_mask = (seg[:, 0:1] > 0.1).float()  # hand is red, obj is blue
        obj_mask = (seg[:, 2:3] > 0.5).float()
        image_utils.save_images(hand_mask, osp.join(save_dir, '%s/hand_mask' % (save_index),  '%05d' % (cnt - 1)))
        image_utils.save_images(obj_mask, osp.join(save_dir, '%s/obj_mask' % (save_index),  '%05d' % (cnt - 1)))

        camera_dict['cTw'].append(cTw.cpu().detach().numpy()[0])
        camera_dict['wTh'].append(wTh.cpu().detach().numpy()[0])
        camera_dict['K_pix'].append(intr.cpu().detach().numpy()[0])
        camera_dict['hTo'].append(hTo.cpu().detach().numpy()[0])
        camera_dict['onTo'].append(onTo.cpu().detach().numpy()[0])
        hand_dict['hA'].append(hA.cpu().detach().numpy()[0])
        hand_dict['beta'].append(beta.cpu().detach().numpy()[0])

    imageio.mimsave(osp.join(save_dir, save_index, 'image.gif'), image_list)
    print('save to', osp.join(save_dir, save_index, 'image.gif'))

    for k, v in camera_dict.items():
        camera_dict[k] = np.array(v)
        print(k, camera_dict[k].shape)
    for k, v in hand_dict.items():
        hand_dict[k] = np.array(v)
        print(hand_dict[k].shape)
    np.savez_compressed(osp.join(save_dir, save_index, 'cameras_hoi.npz'), **camera_dict)
    np.savez_compressed(osp.join(save_dir, save_index, 'hands.npz'), **hand_dict)


def get_cHand(anno, hand_wrapper: ManopthWrapper):
    shape = torch.FloatTensor(anno['handBeta'][None])
    pose = torch.FloatTensor(anno['handPose'][None])  # handTrans
    trans = torch.FloatTensor(anno['handTrans'][None])

    align_rot = torch.FloatTensor([[[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]])
    cTw = geom_utils.rt_to_homo(align_rot, )
    cHand, cJoints = hand_wrapper(cTw.to(device), pose.to(device), trans=trans.to(device), th_betas=shape.to(device))

    cam_intr = torch.FloatTensor(anno['camMat'][None])
    cCorner = mesh_utils.apply_transform(torch.FloatTensor(anno['objCorners3D'][None]), cTw)
    bboxj2d = minmax(torch.cat([proj3d(cJoints.cpu(), cam_intr), proj3d(cCorner, cam_intr)], dim=1))
    bbox_sq = square_bbox(bboxj2d, pad=0.1)

    return cHand, bbox_sq



def get_crop(anno, hand_wrapper):
    shape = torch.FloatTensor(anno['handBeta'][None])
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

    cHand, cJoints = hand_wrapper(geom_utils.matrix_to_se3(cTw @ wTh).to(device), hA.to(device), th_betas=shape.to(device))

    hHand, _ = hand_wrapper(None, hA.to(device))
    cam_intr = torch.FloatTensor(anno['camMat'][None])

    cCorner = mesh_utils.apply_transform(torch.FloatTensor(anno['objCorners3D'][None]), cTw)
    bboxj2d = minmax(torch.cat([proj3d(cJoints.cpu(), cam_intr), proj3d(cCorner, cam_intr)], dim=1))
    bbox_sq = square_bbox(bboxj2d, pad=0.1)

    cam_intr = image_utils.crop_cam_intr(cam_intr, bbox_sq, (H, W))    
    return hTo, cTh, hA, hHand.cpu(), bbox_sq, cam_intr, shape


def get_K(cam, H, W):
    """
    Args:
        cam ([type]): [description]
        H ([type]): [description]
        W ([type]): [description]
    Returns:
        [type]: (N, 2) (N, 2) in NDC??
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

def link():
    data_idr = '/glusterfs/yufeiy2/fair/ihoi_data/HO3D/train'
    vid_list = glob(data_dir, '*')
    for vid in vid_list:
        vid_list

if __name__ == '__main__':
    skip=True
    # link()
        # render_from_original()
    for dt in [10, 5, 2]:
        render('MDF10', 1000, dt) # drill
        # render('SMu1', 651, 1)

        
        # render('SMu41', 0, 10)
        # sugar box
        render('SMu1', 650, dt)
        render('SS2', 0, dt)

        # # 006_mustard_bottle
        render('SM2', 0, dt)
        # # scissor
        # render('GSF11', 0, 10)
        # render('GSF11', 1000, 10)
        # # 003_cracker_box
        # render('MC2', 0, 10)
        # # banana
        # render('BB12', 0, 10)

        # render('AP12', 50, 10, split='evaluation')
        # AP12/0051
        # flow()