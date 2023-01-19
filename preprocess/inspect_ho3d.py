from tqdm import tqdm
import argparse
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
vis_dir = '/home/yufeiy2/scratch/result/vis/'
shape_dir = '/home/yufeiy2/scratch/data/YCB/models'
device = 'cuda:0'

time_len = 30
H = W = 224



def render_amodal_batch(list_file):
    hand_wrapper = ManopthWrapper().to(device)
    with open(osp.join(data_dir, 'Sets/%s.txt' % list_file)) as fp:
        frame_dict = [line.strip() for line in fp] # 66,034
    np.random.seed(123)
    np.random.shuffle(frame_dict)

    for i, index in tqdm(enumerate(frame_dict), total=len(frame_dict)):
        split, vid_index, frame_index = index.split('/')
        # vid_index = osp.dirname(index)
        # frame_index = osp.basename(index)
        save_index = osp.join(save_dir, '{}', f'{vid_index}_{frame_index}_origin')
        done_file = osp.join(save_dir, 'tmp/done', f'{vid_index}_{frame_index}_origin')
        lock_file = osp.join(save_dir, 'tmp/lock', f'{vid_index}_{frame_index}_origin')
        if args.skip and osp.exists(done_file):
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if args.skip:
                continue
        
        render_amodal(split, vid_index, frame_index, save_index, hand_wrapper, True)

        save_index = osp.join(save_dir, '{}', f'{vid_index}_{frame_index}_novel')
        render_amodal(split, vid_index, frame_index, save_index, hand_wrapper, False)

        os.makedirs(done_file, exist_ok=True)
        if args.num > 0 and  i >= args.num:
            break


def render_amodal(split, vid_index, frame_index, save_index, hand_wrapper, gt_camera=True):
    with open(osp.join(data_dir, '%s/%s/meta/%s.pkl' % (split, vid_index, frame_index)), 'rb') as fp:
        anno = pickle.load(fp)
    hTo, cTh, hA, hHand, bbox_sq, cam_intr_crop, beta = get_crop(anno, hand_wrapper, H, W)
    if not gt_camera:
        cTh = mesh_utils.sample_camera_extr_like(cTw=cTh, t_std=0.1)
        cam_intr_crop[:, 0, 2] = W / 2
        cam_intr_crop[:, 1, 2] = H / 2
    mesh_file = osp.join(shape_dir, anno['objName'], 'textured_simple.obj')
    oMesh = mesh_utils.load_mesh(mesh_file)
    hObj = mesh_utils.apply_transform(oMesh, hTo)

    iHand, iObj = render_amodal_from_camera(hHand, hObj, cTh, cam_intr_crop, H, W)
    
    # save everything
    image_utils.save_images(
        torch.cat([iHand['mask'], iObj['mask'], torch.zeros_like(iObj['mask']) ], 1), 
        save_index.format('amodal_mask'))

    save_depth(iHand['depth'], save_index.format('hand_depth'))
    save_depth(iObj['depth'], save_index.format('obj_depth'))

    os.makedirs(osp.dirname(save_index.format('hand_normal')), exist_ok=True)
    np.save(save_index.format('hand_normal'), iHand['normal'][0].cpu().detach().numpy())  # (3, H, W)
    os.makedirs(osp.dirname(save_index.format('obj_normal')), exist_ok=True)
    np.save(save_index.format('obj_normal'), iObj['normal'][0].cpu().detach().numpy())

    image_utils.save_images(iHand['normal'], osp.join(vis_dir, f'{vid_index}_{frame_index}'), scale=True)
    

def save_depth(image, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    image = image.cpu().detach()
    image = image[0].permute([1, 2, 0]).numpy()[..., 0]
    # image = np.clip(image, 0, 1<<8-1)
    # image = image.astype(np.uint8)
    image = np.clip(image, 0, 1<<16-1)
    image = image.astype(np.uint16)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    image = Image.fromarray(image)
    image.save(save_path + '.png')


def render_amodal_from_camera(hHand, hObj, cTh, cam_intr, H, W):
    hHand = hHand.to(device)
    hObj = hObj.to(device)
    cTh = cTh.to(device)

    cHand = mesh_utils.apply_transform(hHand, cTh)
    cObj = mesh_utils.apply_transform(hObj, cTh)
    f, p = mesh_utils.get_fxfy_pxpy(mesh_utils.intr_from_screen_to_ndc(cam_intr, H, W))
    cameras = PerspectiveCameras(f, p).to(device)
    
    iHand = mesh_utils.render_mesh(cHand, cameras, 
        depth_mode=args.depth, 
        normal_mode=args.normal,
        out_size=max(H, W))
    iHand['depth'] *= iHand['mask']
    iObj = mesh_utils.render_mesh(cObj, cameras, 
        depth_mode=args.depth, 
        normal_mode=args.normal,
        out_size=max(H, W))
    iObj['depth'] *= iObj['mask']

    # rescale to [0, 1] to be consistent with midas convention
    iHand['normal'] = iHand['normal'] / 2 + 0.5
    iObj['normal'] = iObj['normal'] / 2 + 0.5

    # sustract common min depth and scale to avoid overflow
    min_depth_hand = torch.min(torch.masked_select(iHand['depth'], iHand['mask'].bool()))
    min_depth_obj = torch.min(torch.masked_select(iObj['depth'], iObj['mask'].bool()))
    min_depth = min(min_depth_hand, min_depth_obj)
        # torch.masked_select(iHand['depth'], iHand['mask'].bool()).min(), 
        # torch.masked_select(iObj['depth'], iObj['mask'].bool()).min()) - 2/1000
    # print('norm', torch.norm(iHand['normal'], -1))
    iHand['depth'] *= 1000
    iObj['depth']  *= 1000   
    iHand['depth'] -= min_depth * 1000  # in mm
    iObj['depth'] -= min_depth * 1000
    return iHand, iObj




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


        hTo, cTh, hA, hHand, bbox_sq, cam_intr_crop, beta = get_crop(anno, hand_wrapper, H, W)

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



def get_crop(anno, hand_wrapper, H, W):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", action='store_true')
    parser.add_argument("--normal", action='store_true')
    parser.add_argument("--skip", action='store_true')

    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--split", type=str, default='train_seg')
    args, unknown = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    skip = args.skip

    save_dir = '/home/yufeiy2/scratch/data/HO3D/crop_render'
    render_amodal_batch(args.split)

    # link()
        # render_from_original()
    # for dt in [10, 5, 2]:
        # render('MDF10', 1000, dt) # drill
        # render('SMu1', 651, 1)

        
        # render('SMu41', 0, 10)
        # sugar box
        # render('SMu1', 650, dt)
        # render('SS2', 0, dt)

        # # 006_mustard_bottle
        # render('SM2', 1, dt)
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