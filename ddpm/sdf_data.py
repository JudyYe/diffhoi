import pickle
import os
from time import time

from utils.hand_utils import ManopthWrapper, get_nTh
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import os.path as osp
from mesh_to_sdf import mesh_to_sdf
import trimesh
from jutils import mesh_utils, image_utils, geom_utils

shapenet_dir = '/glusterfs/yufeiy2/download_data/ShapeNetCore.v2/'
vis_dir = '/glusterfs/yufeiy2/vhoi/vis/'
data_dir = '/glusterfs/yufeiy2/fair/data/obman/'
device = 'cuda:0'


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
    s = 1 / meta_info['obj_scale']
    cTo = torch.FloatTensor([np.matmul(meta_info['cTo'], np.diag([s, s, s, 1]))]).to(device)
    cTh = torch.FloatTensor([meta_info['cTh']]).to(device)
    hTc = geom_utils.inverse_rt(mat=cTh, return_mat=True).to(device)
    hTo = torch.matmul(hTc, cTo).to(device)

    # get hand mesh
    hA = torch.FloatTensor([meta_info['hA']]).to(device)
    
    nTh = get_nTh(hA=hA, hand_wrapper=hand_wrapper,r=0.1)
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
    oMesh = mesh_utils.load_mesh(fname, scale_verts=meta_info['obj_scale']).to(device)
    nMeshes = mesh_utils.apply_transform(oMesh, nTo)
    nMesh = trimesh.Trimesh(
        nMeshes.verts_packed().cpu().numpy(), 
        nMeshes.faces_packed().cpu().numpy())

    nSdf = mesh_to_sdf(nMesh, nPoints.reshape([-1, 3])).reshape(reso, reso, reso)
    
    hHand, _ = hand_wrapper(None, hA)
    nHand = mesh_utils.apply_transform(hHand, nTh)

    return nSdf, nPoints, meta_info['hA'], nHand, nMeshes


def main():
    index = '%08d' % 6755
    hand_wrapper = ManopthWrapper().to(device)

    nSdf, nPoints, hA, nHand, nObj = preprocess_obman(index, hand_wrapper, 'test',)
    # vis 
    nSdf_mesh = vis_sdf(nSdf, None)
    nHoi = mesh_utils.join_scene([nSdf_mesh, nHand])
    vis_mesh(nHoi, osp.join(vis_dir, index))

    nHoi = mesh_utils.join_scene([nObj, nHand])
    vis_mesh(nHoi, osp.join(vis_dir, index) + '_mesh')




def vis():
    shape_file = '/glusterfs/yufeiy2/fair/mesh_sdf/MeshInp/ho3d/all/006_mustard_bottle.obj'
    reso = 32
    index = osp.basename(shape_file).split('.')[0]
    
    mesh_gt = mesh_utils.load_mesh(shape_file)
    mesh_gt, _ = mesh_utils.center_norm_geom(mesh_gt, 0)
    vis_mesh(mesh_gt, osp.join(vis_dir, index) + '_mesh')
    print(mesh_gt.verts_padded().min(), mesh_gt.verts_padded().max())

    mesh = trimesh.load(osp.join(vis_dir, index) + '_mesh.obj')
    x, y, z = torch.meshgrid([
        torch.linspace(-1, 1, reso), 
        torch.linspace(-1, 1, reso), 
        torch.linspace(-1, 1, reso)]
    )  # D, H, W
    print(x.shape)
    query = torch.stack([x, y, z], -1) 
    query = query.detach().cpu().numpy()
    start = time()
    sdf = mesh_to_sdf(mesh, query.reshape(-1, 3), ).reshape([reso, reso, reso])
    print(time() - start, 's')

    vis_sdf(sdf, osp.join(vis_dir, index))

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



if __name__ == '__main__':
    # vis()
    main()