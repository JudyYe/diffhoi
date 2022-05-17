import json
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
from tqdm import tqdm
from ddpm.data import unpack_sdf_samples
from torchmetrics import ScaleInvariantSignalNoiseRatio
from models.frameworks.volsdf_hoi import MeshRenderer

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
time_len = 30
H = W = 512
def convert_to_DTU():
    return 

def render_masks(wHoi, cTw_list, focal, save_dir):
    wHoi = wHoi.to(device)
    cTw_list = cTw_list.to(device)

    for t in range(len(cTw_list)):
        print(t, len(cTw_list))
        c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        cam = PerspectiveCameras(focal, R=c1Tw[0].transpose(-1, -2), T=c1Tw[1], device=device)

        # cHoi = mesh_utils.apply_transform(wHoi, cTw)
        image = mesh_utils.render_mesh(wHoi, cam, out_size=H)

        hand_mask = ((image['image'] * image['mask'])[:, 0:1] > 0.5).float()
        obj_mask = ((image['image'] * image['mask'])[:, 1:2] > 0.5).float()
        image_utils.save_images(hand_mask, osp.join(save_dir, 'hand_mask/%05d' % t))
        image_utils.save_images(obj_mask, osp.join(save_dir, 'obj_mask/%05d' % t))


def render_all(wHoi, cTw_list, focal, save_dir):
    wHoi = wHoi.to(device)
    cTw_list = cTw_list.to(device)

    os.makedirs(osp.join(save_dir, 'FlowFW'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'FlowBW'), exist_ok=True)
    image_list = []
    for t in range(len(cTw_list)):
        print(t, len(cTw_list))
        c1Hoi = mesh_utils.apply_transform(wHoi, cTw_list[t:t+1])
        cam = PerspectiveCameras(focal, device=device)
        image = mesh_utils.render_mesh(c1Hoi, cam, out_size=H)

        # c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        # cam = PerspectiveCameras(focal, R=c1Tw[0].transpose(-1, -2), T=c1Tw[1], device=device)

        # cHoi = mesh_utils.apply_transform(wHoi, cTw)
        # image = mesh_utils.render_mesh(wHoi, cam, out_size=H)
        img_np = image_utils.save_images(image['image'], osp.join(save_dir, 'image/%05d' % t))
        image_utils.save_images(image['mask'], osp.join(save_dir, 'mask/%05d' % t))
        image_list.append(img_np)
    imageio.mimsave(osp.join(save_dir, 'image.gif'), image_list)
    print('save to', osp.join(save_dir, 'image.gif'))

    for t in range(len(cTw_list) -1):
        c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        c2Tw = geom_utils.homo_to_rt(cTw_list[t+1:t+2])
        cam1 = PerspectiveCameras(focal, R=c1Tw[0].transpose(-1, -2), T=c1Tw[1], device=device)
        cam2 = PerspectiveCameras(focal, R=c2Tw[0].transpose(-1, -2), T=c2Tw[1], device=device)

        flow = mesh_utils.render_mesh_flow(wHoi, cam1, cam2, return_ndc=False, out_size=H)

        flow12 = flow['flow'].cpu().detach().numpy()[0]
        flow_image = image_utils.flow_uv_to_colors(flow12[..., 0] / W, flow12[..., 1] / H)

        imageio.imwrite(osp.join(save_dir, 'FlowFW/%05d.png' % t), flow_image)
        np.savez_compressed(osp.join(save_dir, 'FlowFW/%05d.npz' % t), flow=flow12)

    for t in range(1, len(cTw_list)):
        c1Tw = geom_utils.homo_to_rt(cTw_list[t:t+1])
        c2Tw = geom_utils.homo_to_rt(cTw_list[t-1:t])
        cam1 = PerspectiveCameras(focal, R=c1Tw[0].transpose(-1, -2), T=c1Tw[1], device=device)
        cam2 = PerspectiveCameras(focal, R=c2Tw[0].transpose(-1, -2), T=c2Tw[1], device=device)

        flow = mesh_utils.render_mesh_flow(wHoi, cam1, cam2, return_ndc=False, out_size=H)

        flow12 = flow['flow'].cpu().detach().numpy()[0]
        flow_image = image_utils.flow_uv_to_colors(flow12[..., 0] / W, flow12[..., 1] / H)

        imageio.imwrite(osp.join(save_dir, 'FlowBW/%05d.png' % t), flow_image)
        np.savez_compressed(osp.join(save_dir, 'FlowBW/%05d.npz' % t), flow=flow12)


def render(index, split='test'):
    hand_wrapper = ManopthWrapper().to(device)
    hObj, hHand, _ = meta_to_mesh(hand_wrapper, index, split)
    hObj.textures = mesh_utils.pad_texture(hObj, 'blue')
    hHand.textures = mesh_utils.pad_texture(hHand, 'yellow')
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)

    wHoi, wTh = mesh_utils.center_norm_geom(hHoi, 0)
    wHand = mesh_utils.apply_transform(hHand.to(device), wTh)
    wObj = mesh_utils.apply_transform(hObj.to(device), wTh)
    mesh_utils.dump_meshes([osp.join(save_dir, index, 'gt')], wHoi)
    mesh_utils.dump_meshes([osp.join(save_dir, index, 'hand')], wHand)

    azel = mesh_utils.get_view_list('az', device, time_len)  # (T, 2)
    focal = 3.75
    cTw_rot, cTw_trans = look_at_view_transform(2*focal, azel[..., 1], azel[..., 0], False, )
    cTw_list = geom_utils.rt_to_homo(cTw_rot, cTw_trans)
    print(cTw_list[0])
    rad = 2 * focal
    mat = geom_utils.inverse_rt(mat=cTw_list, return_mat=True)[0]
    c2w_tracks = render_view.c2w_track_spiral(
        mat.cpu().detach().numpy(),
        np.array([0., 1., 0.]),
        rads=np.array([0.5* rad,  rad, 0. * rad]), 
        focus=rad, zrate=0.1, rots=3, N=time_len,
    )
    c2w_tracks = torch.FloatTensor(c2w_tracks)
    c2w_tracks = geom_utils.inverse_rt(mat=c2w_tracks,return_mat=True)
    # render_all(wHoi, cTw_list, focal, osp.join(save_dir, index))
    wHoi_label = mesh_utils.join_scene_w_labels([wHand, wObj], 3)
    render_masks(wHoi_label, cTw_list, focal, osp.join(save_dir, index))

    # save camera
    proj_mat = get_proj_intr_mat(cTw_list, H, W, focal)   # T, 4, 4
    camera_dict = {}
    for t in range(time_len): 
        # verts = wHoi.verts_packed().cpu().detach().numpy()  #  P, 3
        # uv = proj(verts, proj_mat[t].cpu().detach().numpy())
        # image = cv2.imread(osp.join(save_dir, index, 'image/%05d.png' % t))

        # draw(uv, image)
        # os.makedirs(osp.join(save_dir, index, 'vis'), exist_ok=True)
        # plt.savefig(osp.join(save_dir, index, 'vis/%d.png' %  t) )

        camera_dict['scale_mat_%d' % t] = np.eye(4)
        camera_dict['world_mat_%d' % t] = proj_mat[t].cpu().detach().numpy()
    np.savez_compressed(osp.join(save_dir, index, 'cameras.npz'), **camera_dict)

    return 

def render_save_for_cTh(index, split='test'):
    hand_wrapper = ManopthWrapper().to(device)
    
    hObj, hHand, hA = meta_to_mesh(hand_wrapper, index, split)
    hObj.textures = mesh_utils.pad_texture(hObj, 'blue')
    hHand.textures = mesh_utils.pad_texture(hHand, 'yellow')
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)

    wHoi, wTh = mesh_utils.center_norm_geom(hHoi, 0)
    wHand = mesh_utils.apply_transform(hHand.to(device), wTh)
    wObj = mesh_utils.apply_transform(hObj.to(device), wTh)
    wCenter = wHand.verts_packed().mean(0, keepdim=True)
    print(hHand.verts_packed().mean(0)) 


    azel = mesh_utils.get_view_list('az', device, time_len)  # (T, 2)
    focal = 3.75
    cTw_rot, cTw_trans = look_at_view_transform(2*focal, azel[..., 1], azel[..., 0], False, at=wCenter)
    cTw_list = geom_utils.rt_to_homo(cTw_rot, cTw_trans)
    print(cTw_list[0])
    
    # change w to H  # toulianghuanzhu
    # wHand = hHand
    # wHoi = hHoi
    # wObj = hObj
    # cTw_list = cTw_list @ wTh.get_matrix().transpose(-1, -2).cpu()

    rot, trans, scale = geom_utils.homo_to_rt(wTh.get_matrix().transpose(-1, -2).to(device))
    wTsh = geom_utils.rt_to_homo(rot, trans)
    shTh = geom_utils.scale_matrix(scale)
    cTw_list = cTw_list.to(device) @ wTsh
    wHoi = mesh_utils.apply_transform(hHoi, shTh)
    wHand = mesh_utils.apply_transform(hHand.to(device), shTh)
    wObj = mesh_utils.apply_transform(hObj.to(device), shTh)

    mesh_utils.dump_meshes([osp.join(save_dir, index + '_cTh', 'gt')], wHoi)
    mesh_utils.dump_meshes([osp.join(save_dir, index + '_cTh', 'hand')], wHand)

    render_all(wHoi, cTw_list, focal, osp.join(save_dir, index + '_cTh'))
    wHoi_label = mesh_utils.join_scene_w_labels([wHand, wObj], 3)
    render_masks(wHoi_label, cTw_list, focal, osp.join(save_dir, index + '_cTh'))

    # save camera
    intr_mat = get_intr_mat(H, W, focal)
    camera_dict = {}
    # for t in range(time_len): 
        # verts = wHoi.verts_packed().cpu().detach().numpy()  #  P, 3
        # uv = proj(verts, proj_mat[t].cpu().detach().numpy())
        # image = cv2.imread(osp.join(save_dir, index, 'image/%05d.png' % t))

        # draw(uv, image)
        # os.makedirs(osp.join(save_dir, index, 'vis'), exist_ok=True)
        # plt.savefig(osp.join(save_dir, index, 'vis/%d.png' %  t) )

    camera_dict['cTw'] = cTw_list.cpu().detach().numpy()
    camera_dict['wTh'] = np.tile(shTh.cpu().detach().numpy(), [time_len, 1, 1])  # scale matrix
    camera_dict['K_pix'] = np.tile(intr_mat.cpu().detach().numpy(), [time_len, 1, 1])
    camera_dict['hTo'] =np.tile(np.eye(4)[None], [time_len, 1, 1])

    np.savez_compressed(osp.join(save_dir, index + '_cTh', 'cameras_hoi.npz'), **camera_dict)

    hand_dict = {}
    hand_dict['hA'] = np.tile(hA.cpu().detach().numpy(), [time_len, 1, 1])
    np.savez_compressed(osp.join(save_dir, index + '_cTh', 'hands.npz'), **hand_dict)

    return 


def cv2_scatter(vp, image, color):
    for v in vp:
        cv2.circle(image, (v[0], v[1]), 1, color)
    return image

def get_proj_intr_mat(cTw, H, W, focal, px=0, py=0):
    cali = torch.eye(4)
    # cali[0, 0] = -1
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


def get_intr_mat(H, W, focal, px=0, py=0):
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
    proj = scale_intr @ intr_ndc
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

    return hObj, hHand, hA

# def get_default_camera_traj():
#     render_view.c2w_track_spiral(
#         c2w, [0, 0, 1], [0.1, 2, 3], 
#         (0, 0, 20), 1, 1, 
#     )

def rerender():
    hand_wrapper = ManopthWrapper().to(device)
    mesh_renderer = MeshRenderer().to(device)

    instance_dir = '/checkpoint/yufeiy2/vhoi_out/syn_data/00006755_cTh'
    hands = np.load('{0}/hands.npz'.format(instance_dir))
    hA = torch.from_numpy(hands['hA']).float().squeeze(1)[0:1]

    cam_file = '{0}/cameras_hoi.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    wTc = geom_utils.inverse_rt(mat=torch.from_numpy(camera_dict['cTw']).float(), return_mat=True)[0:1].to(device)

    wTh = torch.from_numpy(camera_dict['wTh']).float()[0:1].to(device)
    hTw = geom_utils.inverse_rt(mat=wTh, return_mat=True)
    hTo =  torch.from_numpy(camera_dict['hTo']).float()[0:1].to(device)
    intrinsics = torch.from_numpy(camera_dict['K_pix']).float()[0:1].to(device)

    oTh = geom_utils.inverse_rt(mat=hTo, return_mat=True)
    oTc = oTh @ hTw @ wTc  
    
    print(torch.det(wTc[..., 0:3, 0:3]), torch.det(oTc[..., 0:3, 0:3]))

    print(intrinsics)
    hHand, _ = hand_wrapper(None, hA.cuda())
    wHand = mesh_utils.apply_transform(hHand, wTh)
    iHand = mesh_renderer(
        geom_utils.inverse_rt(mat=wTc, return_mat=True), intrinsics, wHand, H=H, W=W, far=100,
    )
    # oHand = mesh_utils.apply_transform(hHand, oTh)
    # iHand = mesh_renderer(
    #     geom_utils.inverse_rt(mat=oTc, return_mat=True), intrinsics, oHand, H=H, W=W, far=100,
    # )

    mesh_utils.dump_meshes([osp.join(save_dir, 'tmp/hHand')], hHand)
    image_utils.save_images(iHand['image'], osp.join(save_dir, 'tmp/image'))


def parser_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default='%08d'%6755, type=str)
    return parser
if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    args = parser_default().parse_args()

    # render(args.seq)
    # render_save_for_cTh(args.seq)
    # rerender()
    osdf()