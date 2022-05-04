import argparse
import json
import logging
import imageio
import numpy as np
import pickle
import os
from time import time

from tqdm import tqdm
from ddpm.mow import load_mow

from utils.hand_utils import ManopthWrapper, get_nTh
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import os.path as osp
from mesh_to_sdf import mesh_to_sdf
import trimesh
from jutils import mesh_utils, image_utils, geom_utils

vis_dir = '/glusterfs/yufeiy2/vhoi/vis/'
save_dir = '/glusterfs/yufeiy2/vhoi/obman/grid_sdf/'
shapenet_dir = '/glusterfs/yufeiy2/download_data/ShapeNetCore.v2/'
tight_dir = '/glusterfs/yufeiy2/download_data/ShapeNetCore.v2_water/'
data_dir = '/glusterfs/yufeiy2/fair/data/obman/'
device = 'cuda:0'

def load_obman(index, hand_wrapper, split):
    # split, index = index.split('/')
    anno = os.path.join(data_dir, split, 'meta_plus', index + '.pkl')
    with open(anno, 'rb') as fp:
        meta_info = pickle.load(fp)

    # get hTo
    cTo = torch.FloatTensor([meta_info['cTo']]).to(device)
    cTh = torch.FloatTensor([meta_info['cTh']]).to(device)
    hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True).to(device)
    hTo = torch.matmul(hTc, cTo).to(device)

    # get hand mesh
    hA = torch.FloatTensor([meta_info['hA']]).to(device)

    shape_dir = os.path.join(shapenet_dir, '{}', '{}', 'models', 'model_normalized.obj')
    fname = shape_dir.format(meta_info['class_id'], meta_info['sample_id'])
    fname = water_tight_and_uv(fname, fname.replace('model_normalized.obj', 'water_tight'))
    oMesh = mesh_utils.load_mesh(fname, ).to(device)

    hMesh = mesh_utils.apply_transform(oMesh, hTo)
    return hMesh, hA


def preprocess_obman(index, hand_wrapper, split, reso=32, device='cuda:0'):
    """
    nSDf: numpy() (H, H, H)
    nPoints: numpy (H, H, H, 3)  # grids that spans -1, 1
    hA: numpy() (45()
    """
    # train/00001
    # split, index = index.split('/')
    anno = os.path.join(data_dir, split, 'meta_plus', index + '.pkl')
    with open(anno, 'rb') as fp:
        meta_info = pickle.load(fp)

    # get hTo
    cTo = torch.FloatTensor([meta_info['cTo']]).to(device)
    cTh = torch.FloatTensor([meta_info['cTh']]).to(device)
    hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True).to(device)
    hTo = torch.matmul(hTc, cTo).to(device)

    # get hand mesh
    hA = torch.FloatTensor([meta_info['hA']]).to(device)
    
    nTh = get_nTh(hA=hA, hand_wrapper=hand_wrapper)
    nTo = nTh @ hTo
    oTn = geom_utils.inverse_rt(mat=nTh @ hTo, return_mat=True)

    x, y, z = torch.meshgrid([
        torch.linspace(-1, 1, reso), 
        torch.linspace(-1, 1, reso), 
        torch.linspace(-1, 1, reso)]
    )  # D, H, W
    nPoints = torch.stack([x, y, z], -1).detach().numpy()

    shape_dir = os.path.join(shapenet_dir, '{}', '{}', 'models', 'model_normalized.obj')
    fname = shape_dir.format(meta_info['class_id'], meta_info['sample_id'])
    fname = water_tight_and_uv(fname, fname.replace('model_normalized.obj', 'water_tight'))
    oMesh = mesh_utils.load_mesh(fname, ).to(device)
    nMeshes = mesh_utils.apply_transform(oMesh, nTo)
    nMesh = trimesh.Trimesh(
        nMeshes.verts_packed().cpu().numpy(), 
        nMeshes.faces_packed().cpu().numpy())

    nSdf = mesh_to_sdf(nMesh, nPoints.reshape([-1, 3]), sign_method="depth", scan_count=10000, scan_resolution=8000).reshape(reso, reso, reso)
    
    hHand, _ = hand_wrapper(None, hA)
    nHand = mesh_utils.apply_transform(hHand, nTh)

    return nSdf, nPoints, meta_info['hA'], nHand, nMeshes


def water_tight_and_uv(inp_file, out_file, reso=5000):
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    logging.info('water tight mesh to %s' % out_file)
    out_file += '.obj'
    if not osp.exists(out_file):
        cmd = '/home/yufeiy2/Tools/Manifold/build/manifold %s %s %d' % (inp_file, out_file, reso)
        print(cmd)
        os.system(cmd)
    return out_file


def preprocess(hand_wrapper, hMesh, hA, reso=32):
    nTh = get_nTh(hA=hA, hand_wrapper=hand_wrapper)

    x, y, z = torch.meshgrid([
        torch.linspace(-1, 1, reso), 
        torch.linspace(-1, 1, reso), 
        torch.linspace(-1, 1, reso)]
    )  # D, H, W
    nPoints = torch.stack([x, y, z], -1).detach().numpy()

    nMeshes = mesh_utils.apply_transform(hMesh, nTh)
    nMesh = trimesh.Trimesh(
        nMeshes.verts_packed().cpu().numpy(), 
        nMeshes.faces_packed().cpu().numpy())

    nSdf = mesh_to_sdf(nMesh, nPoints.reshape([-1, 3])).reshape(reso, reso, reso)
    
    hHand, _ = hand_wrapper(None, hA)
    nHand = mesh_utils.apply_transform(hHand, nTh)

    return nSdf, nPoints, hA[0].cpu().detach().numpy(), nHand, nMeshes


def run_one(index, hand_wrapper, hMesh, hA):
    out_file = osp.join(save_dir, index) + '.npz'
    if args.skip and osp.exists(out_file):
        logging.info('skip', out_file)
        return 
    lock = out_file + '.lock'
    try:
        os.makedirs(lock)
    except FileExistsError:
        if args.skip:
            return
    
    # oMesh, hTo, hA = load
    nSdf, nPoints, hA, nHand, nObj = preprocess(hand_wrapper, hMesh, hA)
    # nSdf, nPoints, hA, nHand, nObj = preprocess_obman(index, hand_wrapper, split,)
    
    if args.vis:
        # vis 
        nSdf_mesh = vis_sdf(nSdf, None)
        nHoi = mesh_utils.join_scene([nSdf_mesh, nHand])
        vis_mesh(nHoi, osp.join(vis_dir, index))

        nHoi = mesh_utils.join_scene([nObj, nHand])
        vis_mesh(nHoi, osp.join(vis_dir, index) + '_mesh')

        image_dir = osp.join(data_dir, 'images/{0}.jpg')
        image = imageio.imread(image_dir.format(index))
        imageio.imwrite(osp.join(vis_dir, index) + '.jpg', image)


    # (D, H, W), (D, H, W, 3) (45)
    np.savez_compressed(out_file, 
        nSdf=nSdf, nPoints=nPoints, hA=hA)

    os.system('rm -rf %s' % lock)


def vis_sdf(sdf, mesh_file=None):
    sdf_tensor = torch.FloatTensor(sdf).unsqueeze(0)
    meshes = mesh_utils.batch_grid_to_meshes(sdf_tensor, 1, N=sdf_tensor.shape[-1])
    if mesh_file is not None:
        vis_mesh(meshes, mesh_file)
    return meshes


def vis_mesh(meshes, mesh_file):
    meshes.textures = mesh_utils.pad_texture(meshes)
    mesh_utils.dump_meshes([mesh_file], meshes)
    image_list = mesh_utils.render_geom_rot(meshes.cuda(), scale_geom=True)
    image_utils.save_gif(image_list, mesh_file)



def main(split='test'):
    index_list = [line.strip() for line in open(osp.join(data_dir, split + '.txt'))]
    if args.data == 'mow':
        index_list = [e['image_id'] for e in annos]
    hand_wrapper = ManopthWrapper().to(device)
    idx = 0
    for idx, index in tqdm(enumerate(index_list), total=len(index_list)):
        if args.data == 'obman':
            hMesh, hA = load_obman(index, hand_wrapper, split)
        elif args.data == 'mow':
            hMesh, hA = load_mow(annos[idx], hand_wrapper, data_dir, device)
        # nSdf, nPoints, hA, nHand, nObj = preprocess_obman(index, hand_wrapper, split,)

        run_one(index, hand_wrapper, hMesh, hA)
        if args.num > 0 and idx >= args.num:
            break


def make_mini_list():
    for split in ['test', 'train']:
        index_list = [line.strip() for line in open(osp.join(data_dir, split + '.txt'))]
        with open(osp.join(data_dir, split + '_mini.txt'), 'w') as fp:
            for i in range(100):
                if not osp.exists(osp.join(save_dir, '%s.npz' % index_list[i])):
                    logging.info('skip %s' % osp.join(save_dir, '%s.npz' % index_list[i]))
                    continue
                fp.write('%s\n' % index_list[i])
            
    return 


def make_list():
    for split in ['test', 'train']:
        index_list = [line.strip() for line in open(osp.join(data_dir, split + '.txt'))]
        with open(osp.join(data_dir, split + '_full.txt'), 'w') as fp:
            for i in tqdm(range(len(index_list))):
                if not osp.exists(osp.join(save_dir, '%s.npz' % index_list[i])):
                    logging.info('skip %s' % osp.join(save_dir, '%s.npz' % index_list[i]))
                    continue
                fp.write('%s\n' % index_list[i])
    return 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num', default=-1, type=int)
    parser.add_argument('--data', default='mow', type=str)
    args = parser.parse_args()
    if args.data == 'obman':
        save_dir = '/glusterfs/yufeiy2/vhoi/obman/grid_sdf/'
        shapenet_dir = '/glusterfs/yufeiy2/download_data/ShapeNetCore.v2/'
        tight_dir = '/glusterfs/yufeiy2/download_data/ShapeNetCore.v2_water/'
        data_dir = '/glusterfs/yufeiy2/fair/data/obman/'
    elif args.data == 'mow':
        save_dir = '/glusterfs/yufeiy2/vhoi/mow/grid_sdf/'
        data_dir = '/glusterfs/yufeiy2/download_data/MOW/mow/' 
        with open(data_dir + '/poses.json', 'r') as f:
            annos = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # vis()
    main(args.split)
    # make_mini_list()
    # make_list()     