import argparse
from glob import glob
import logging
from PIL import Image
import cv2
import os
import os.path as osp
import random
import imageio
import numpy as np 
import shutil
import torch
import json
import mano
from pytorch3d.transforms.transform3d import Rotate, Translate, Scale
from pytorch3d.structures.meshes import Meshes
from utils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o
from jutils import image_utils, mesh_utils, geom_utils

data_dir = '/glusterfs/yufeiy2/download_data/MOW/mow/' 
vis_dir = '/glusterfs/yufeiy2/vhoi/vis/'

def center_vertices(vertices, faces, flip_y=True):
    """Centroid-align vertices."""
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces

def water_tight_and_uv(inp_file, out_file, reso=5000):
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    logging.info('water tight mesh to %s' % out_file)
    out_file += '.obj'
    print(out_file)
    if not osp.exists(out_file):
        cmd = '/home/yufeiy2/Tools/Manifold/build/manifold %s %s %d' % (inp_file, out_file, reso)
        print(cmd)
        os.system(cmd)
    return out_file


def load_mow(anno, hand_wrapper, data_dir, device='cuda:0'):
    index = anno['image_id']

    shape_dir = osp.join(data_dir, 'models/{0}.obj')
    tight_dir = osp.join(data_dir, 'tight/{0}')

    mano_pose = torch.tensor(anno['hand_pose']).unsqueeze(0).to(device)
    transl = torch.tensor(anno['trans']).unsqueeze(0).to(device)
    pose, global_orient = mano_pose[:, 3:], mano_pose[:, :3]
    hA = pose + hand_wrapper.hand_mean

    wrTh = geom_utils.axis_angle_t_to_matrix(*cvt_axisang_t_i2o(global_orient, transl))
    wTwr = apply_trans(anno['hand_R'], anno['hand_t'], anno['hand_s']).to(device)
    wTh = wTwr @ wrTh
    
    wTo = apply_trans(anno['R'], anno['t'], anno['s']).to(device)
    verts, faces = load_obj(shape_dir.format(index))
    verts, faces = center_vertices(verts, faces)
    
    fname = tight_dir.format(index)
    oObj = Meshes([verts], [faces]).to(device)
    if not osp.exists(fname + '.obj'):
        mesh_utils.dump_meshes([fname], oObj)
    fname = water_tight_and_uv(fname + '.obj', fname + '_tight')
    oObj = mesh_utils.load_mesh(fname, device=device)

    hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo
    hObj = mesh_utils.apply_transform(oObj, hTo)
    return  hObj, hA


def apply_trans(rot, t, s, device='cpu'):
    trans = Translate(torch.tensor(t).reshape((1, 3)), device=device)
    # rot = Rotate(torch.tensor(rot).reshape((1, 3, 3)).transpose(-1, -2), device=device)
    rot = Rotate(torch.tensor(rot).reshape((1, 3, 3)), device=device)
    scale = Scale(s).to(device)

    chain = rot.compose(trans, scale)
    mat = chain.get_matrix().transpose(-1, -2)
    return mat


def get_one(anno, hand_wrapper, root='zero', device='cuda:0'):
    index = anno['image_id']

    image_dir = osp.join(data_dir, 'images/{0}.jpg')
    shape_dir = osp.join(data_dir, 'models/{0}.obj')
    image = imageio.imread(image_dir.format(index))
    imageio.imwrite(osp.join(vis_dir, index) + '.jpg', image)

    mano_pose = torch.tensor(anno['hand_pose']).unsqueeze(0).to(device)
    transl = torch.tensor(anno['trans']).unsqueeze(0).to(device)
    pose, global_orient = mano_pose[:, 3:], mano_pose[:, :3]
    pose = pose + hand_wrapper.hand_mean

    wrTh = geom_utils.axis_angle_t_to_matrix(*cvt_axisang_t_i2o(global_orient, transl))
    wTwr = apply_trans(anno['hand_R'], anno['hand_t'], anno['hand_s']).to(device)
    wTh = wTwr @ wrTh
    wHand, _ = hand_wrapper(wTh, pose)
    
    wTo = apply_trans(anno['R'], anno['t'], anno['s']).to(device)
    verts, faces = load_obj(shape_dir.format(index))
    verts, faces = center_vertices(verts, faces)
    oObj = Meshes([verts], [faces]).to(device)

    wObj = mesh_utils.apply_transform(oObj, wTo)
    return  wHand, wObj


def make_list():
    image_dir = osp.join(data_dir, 'images/{0}.jpg')
    image_list = glob(image_dir.format('*'))
    index_list = [osp.basename(e).split('.')[0] for e in image_list]
    with open(osp.join(data_dir, 'all.txt'), 'w') as fp:
        for index in index_list:
            fp.write('%s\n' % index)
    np.random.seed(123)
    np.random.permutation(index_list)
    num = len(index_list) // 10
    with open(osp.join(data_dir, 'test_full.txt'), 'w') as fp:
        for index in index_list[:num]:
            fp.write('%s\n' % index)
    with open(osp.join(data_dir, 'train_full.txt'), 'w') as fp:
        for index in index_list[num:]:
            fp.write('%s\n' % index)

def main():
    device = 'cuda:0'
    with open(data_dir + '/poses.json', 'r') as f:
        annos = json.load(f)
    
    anno = annos[idx]
    hand_wrapper = ManopthWrapper().to(device)
    hand, obj = get_one(anno, hand_wrapper, device=device)
    hoi = mesh_utils.join_scene([hand, obj])
    image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
    # image_utils.save_gif(image_list, osp.join(vis_dir, anno['image_id']))
    image_utils.save_gif(image_list, osp.join(vis_dir, '%04d' % idx))



def load_obj(filename_obj, normalization=True, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
    textures = None

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=1)
    args = parser.parse_args()
    
    idx = args.idx
    # main()
    make_list()