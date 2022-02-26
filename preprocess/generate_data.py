import imageio
import cv2
import numpy as np
import os
import os.path as osp
import pickle
import argparse
from pyparsing import FollowedBy
import torch
import pytorch3d
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.structures import Pointclouds
from torchmetrics import ScaleInvariantSignalNoiseRatio
from tools import render_view
from jutils import image_utils, mesh_utils, geom_utils
from utils.hand_utils import ManopthWrapper
from utils.rend_util import load_K_Rt_from_P
import matplotlib
import matplotlib.pyplot as plt

"""render flow, rgb, mask of obman in DTU format"""
shapenet_dir = '/checkpoint/yufeiy2/datasets/ShapeNetCore.v2.mirror/'
data_dir = '/checkpoint/yufeiy2/datasets/obman/obman'
save_dir = '/checkpoint/yufeiy2/vhoi_out/syn_data/'
device = 'cuda:0'
time_len = 5
H = W = 512
def convert_to_DTU():
    return 

def render_all(wHoi, cTw_list, focal, save_dir):
    wHoi = wHoi.to(device)
    cTw_list = cTw_list.to(device)

    os.makedirs(osp.join(save_dir, 'FlowFW'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'FlowBW'), exist_ok=True)
    for t in range(len(cTw_list)):
        c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        cam = PerspectiveCameras(focal, R=c1Tw[0], T=c1Tw[1], device=device)

        # cHoi = mesh_utils.apply_transform(wHoi, cTw)
        image = mesh_utils.render_mesh(wHoi, cam, out_size=H)
        image_utils.save_images(image['image'], osp.join(save_dir, 'image/%05d' % t))
        image_utils.save_images(image['mask'], osp.join(save_dir, 'mask/%05d' % t))

    for t in range(len(cTw_list) -1):
        c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        c2Tw = geom_utils.homo_to_rt(cTw_list[t+1:t+2])
        cam1 = PerspectiveCameras(focal, R=c1Tw[0], T=c1Tw[1], device=device)
        cam2 = PerspectiveCameras(focal, R=c2Tw[0], T=c2Tw[1], device=device)

        flow = mesh_utils.render_mesh_flow(wHoi, cam1, cam2, return_ndc=False, out_size=H)

        flow12 = flow['flow'].cpu().detach().numpy()[0]
        flow_image = image_utils.flow_uv_to_colors(flow12[..., 0] / W, flow12[..., 1] / H)

        imageio.imwrite(osp.join(save_dir, 'FlowFW/%05d.png' % t), flow_image)
        np.savez_compressed(osp.join(save_dir, 'FlowFW/%05d.npz' % t), flow=flow12)


    for t in range(1, len(cTw_list)):
        c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        c2Tw = geom_utils.homo_to_rt(cTw_list[t-1:t])
        cam1 = PerspectiveCameras(focal, R=c1Tw[0], T=c1Tw[1], device=device)
        cam2 = PerspectiveCameras(focal, R=c2Tw[0], T=c2Tw[1], device=device)

        flow = mesh_utils.render_mesh_flow(wHoi, cam1, cam2, return_ndc=False, out_size=H)

        flow12 = flow['flow'].cpu().detach().numpy()[0]
        flow_image = image_utils.flow_uv_to_colors(flow12[..., 0] / W, flow12[..., 1] / H)

        imageio.imwrite(osp.join(save_dir, 'FlowBW/%05d.png' % t), flow_image)
        np.savez_compressed(osp.join(save_dir, 'FlowBW/%05d.npz' % t), flow=flow12)


def render(index, split='test'):
    hand_wrapper = ManopthWrapper().to(device)
    hObj, hHand = meta_to_mesh(hand_wrapper, index, split)
    hObj.textures = mesh_utils.pad_texture(hObj, 'random')
    hHand.textures = mesh_utils.pad_texture(hHand, 'yellow')
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)

    wHoi = mesh_utils.center_norm_geom(hHoi, 0)
    azel = mesh_utils.get_view_list('az', device, time_len)  # (T, 2)
    focal = 3.75
    cTw_rot, cTw_trans = look_at_view_transform(2*focal, azel[..., 1], azel[..., 0], False, )
    cTw_list = geom_utils.rt_to_homo(cTw_rot, cTw_trans)
 
    render_all(wHoi, cTw_list, focal, osp.join(save_dir, index))

    # save camera
    proj_mat = get_proj_intr_mat(cTw_list, H, W, focal)   # T, 4, 4
    camera_dict = {}
    for t in range(time_len): 
        camera_dict['scale_mat_%d' % t] = np.eye(4)
        camera_dict['world_mat_%d' % t] = proj_mat[t].cpu().detach().numpy()

        P = proj_mat[t].cpu().detach().numpy()
        P = P[:3, :4]
        intr, pose = load_K_Rt_from_P(P)

        # verts = wHoi.verts_packed().cpu().detach().numpy()  #  P, 3
        # ones = np.ones([len(verts), 1])
        # vp = np.concatenate([verts, ones], -1) @ P.T
        # vp = (vp[..., :2] / vp[..., 2:]).astype(int)        

        # os.makedirs(osp.join(save_dir, index, 'vis'), exist_ok=True)

        # image = cv2.imread(osp.join(save_dir, index, 'image/%05d.png' % t))
        # image = cv2_scatter(vp, image, (0, 255, 0))

        # ones = np.ones([len(verts), 1])
        # vp = np.concatenate([verts, ones], -1) @ (intr @ pose).T
        # vp = (vp[..., :2] / vp[..., 2:3]).astype(int)
        # image = cv2_scatter(vp, image, (255, 0, 0))

        # cv2.imwrite(osp.join(save_dir, index, 'vis/%05d.png' % t), image)
    np.savez_compressed(osp.join(save_dir, index, 'cameras.npz'), **camera_dict)
    return 

def cv2_scatter(vp, image, color):
    for v in vp:
        cv2.circle(image, (v[0], v[1]), 1, color)
    return image

def get_proj_intr_mat(cTw, H, W, focal, px=0, py=0):
    cali = torch.eye(4)
    # cali[1, 1] = -1
    # cali[2, 2] = -1
    cali = cali[None]
    intr_ndc = torch.FloatTensor([[
        [focal, 0, px, 0],
        [0, focal, py, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]])
    scale_intr = torch.FloatTensor([[
        [W / 2, 0, W/2, 0],
        [0, H / 2, H/2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]])
    proj = scale_intr @ intr_ndc @ cTw @ cali
    return proj


    

def water_tight_and_uv(inp_file, index, reso=5000):
    out_file = osp.join(save_dir, index + '.obj')
    if not osp.exists(out_file):
        cmd = '/private/home/yufeiy2/Tools/Manifold/build/manifold %s %s %d' % (inp_file, out_file, reso)
        print(cmd)
        os.system(cmd)
    return out_file



def meta_to_mesh(hand_wrapper, index, split):
    """
    :param hand_wrapper:
    :param index:
    :return: Meshes with N = 1
    """
    # train/00001
    # split, index = index.split('/')
    anno = os.path.join(data_dir, split, 'meta_plus', index + '.pkl')
    with open(anno, 'rb') as fp:
        meta_info = pickle.load(fp)

    # get hTo
    s = 1 / meta_info['obj_scale']
    cTo = torch.FloatTensor([np.matmul(meta_info['cTo'], np.diag([s, s, s, 1]))]).to(device)
    cTh = torch.FloatTensor([meta_info['cTh']]).to(device)
    hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True).to(device)
    hTo = torch.matmul(hTc, cTo).to(device)

    # get hand mesh
    hA = torch.FloatTensor([meta_info['hA']]).to(device)
    zeros = torch.zeros([1, 3], device=device)
    hHand, _ = hand_wrapper(None, hA, zeros, mode='inner')

    # get obj mesh
    shape_dir = os.path.join(shapenet_dir, '{}', '{}', 'models', 'model_normalized.obj')
    fname = shape_dir.format(meta_info['class_id'], meta_info['sample_id'])

    # water tight
    fname = water_tight_and_uv(fname, index)    

    # print(fname)
    obj = mesh_utils.load_mesh(fname, scale_verts=meta_info['obj_scale']).to(device)

    hObj = mesh_utils.apply_transform(obj, hTo)

    hObj = hObj.to('cpu')
    hHand = hHand.to('cpu')

    return hObj, hHand

# def get_default_camera_traj():
#     render_view.c2w_track_spiral(
#         c2w, [0, 0, 1], [0.1, 2, 3], 
#         (0, 0, 20), 1, 1, 
#     )

def parser_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default='%08d'%6755, type=str)
    return parser
if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    args = parser_default().parse_args()

    render(args.seq)