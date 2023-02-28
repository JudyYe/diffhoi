
import os
import os.path as osp
import trimesh
import numpy as np
import copy
import torch
from tqdm import tqdm

from chamferdist import ChamferDistance
from pytorch3d.structures import Meshes
from jutils import image_utils, mesh_utils, geom_utils
from jutils.mesh_utils import Meshes


def register_meshes(source: Meshes, target: Meshes, type='icp_common', scale=True):
    device = source.device
    # first normalize bc icp only works with mesh with coarse alignment
    # record the transformation of target
    if scale:
        source, cTo_s = mesh_utils.center_norm_geom(source)
        target, cTo_t = mesh_utils.center_norm_geom(target)
    else:
        # substract 
        source, cTo_s = mesh_utils.center_norm_geom(source, max_norm=None)
        target, cTo_t = mesh_utils.center_norm_geom(target, max_norm=None)

    source_list = mesh_utils.to_trimeshes(source)
    target_list = mesh_utils.to_trimeshes(target)
    new_source_list = []
    new_target_list = []
    for s, t in zip(source_list, target_list):
        s, t = register(s, t, type, scale=scale)
        new_source_list.append(s)
        new_target_list.append(t)
    new_source = mesh_utils.from_trimeshes(new_source_list).to(device)
    new_target = mesh_utils.from_trimeshes(new_target_list).to(device)

    new_target = mesh_utils.apply_transform(new_target, cTo_t.inverse())
    new_source = mesh_utils.apply_transform(new_source, cTo_t.inverse())
    return new_source, new_target


def register(source, target, type='icp_common', scale=True):
    if type == 'icp_common':
        from trimesh.registration import mesh_other
    elif type == 'icp_constrained':
        from tools.icp import mesh_other
    else:
        raise ValueError('Registration Type Should Be in {icp_common} and {icp_constrained}.')

    # register
    source2target, cost = mesh_other(source, target, scale=scale)
    # source2target, cost = mesh_other(source, target, scale=False)

    # transform
    source.apply_transform(source2target)

    return source, target


def compare(source_file, target_file, iters=1, flip_x=False, flip_y=False, flip_z=False, robust_norm_source=False):
    chamferDist = ChamferDistance()
    
    if isinstance(source_file, str):
        source = trimesh.load(source_file)  # reconstructed mesh
    else:
        source = source_file
    if isinstance(target_file, str):
        target = trimesh.load(target_file)  # ground truth mesh
    else:
        target = target_file

    # flip
    if flip_x:
        source.apply_translation([-source.centroid[0], 0, 0])
    if flip_y:
        source.apply_translation([0, -source.centroid[1], 0])
    if flip_z:
        source.apply_translation([0, 0, -source.centroid[2]])

    # normalize
    if robust_norm_source:
        # use 80% of the vertices
        vertices = source.vertices
        print(vertices.shape)
        vertices = vertices[np.argsort(np.linalg.norm(vertices, axis=-1))[:int(vertices.shape[0] * 0.8)]]  # 
        center_mass = np.mean(vertices, axis=0)
        source.vertices -= center_mass

        vertices = source.vertices
        vertices = vertices[np.argsort(np.linalg.norm(vertices, axis=-1))[:int(vertices.shape[0] * 0.8)]]  # 
        max_norm = np.max(np.linalg.norm(vertices, axis=-1))
        source.vertices /= max_norm

    else:
        source.vertices -= source.center_mass
        source.vertices /= source.vertices.max()
    target.vertices -= target.center_mass
    target.vertices /= target.vertices.max()

    for i in range(iters):

        # register
        new_source, _ = register(source, target)
        # new_source = source
        # if args.iter == 1:
        #     pyrender_vis([new_source, target])

        # chamfer distance 
        vertices_source = new_source.vertices
        vertices_target = target.vertices
        # todo: change to sample point, not vertices... 
        vertices_source = torch.tensor(vertices_source, dtype=torch.float32)[None].cuda()
        vertices_target = torch.tensor(vertices_target, dtype=torch.float32)[None].cuda()
        dist_bidirectional = chamferDist(vertices_source, vertices_target, bidirectional=True) * 0.001
        print(dist_bidirectional.detach().cpu().item())
    
    return dist_bidirectional, (vertices_source, source.faces), (vertices_target, target.faces)

def test():
    data_dir = '/home/yufeiy2/scratch/result/vis'
    device = 'cuda:0'
    source_file = osp.join(data_dir, 'a.obj')
    target_file = osp.join(data_dir, 'b.ply')

    source = trimesh.load(source_file, force='mesh')  # reconstructed mesh
    target = trimesh.load(target_file, force='mesh')  # ground truth mesh

    # flip

    # normalize
    source.vertices -= source.center_mass
    source.vertices /= source.vertices.max()
    target.vertices -= target.center_mass
    target.vertices /= target.vertices.max()
    
    mesh_f = Meshes(
        [torch.FloatTensor(target.vertices)], 
        [torch.LongTensor(target.faces)]).to(device)
    mesh_s = Meshes(
        [torch.FloatTensor(source.vertices)], 
        [torch.LongTensor(source.faces)]).to(device)
    hoi = mesh_utils.join_scene_w_labels([mesh_s, mesh_f])
    image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(data_dir, 'before'))

    for i in range(args.iter):
        # register
        new_s, _ = register_meshes(mesh_s, mesh_f)
    
        hoi = mesh_utils.join_scene_w_labels([new_s, mesh_f])
        image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(data_dir, 'align_%d' % i))
        mesh_utils.dump_meshes([osp.join(data_dir, 'align_%d' % i)], mesh_s)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/yufeiy2/scratch/result/vhoi/pred_calib/Mug_1_0_len1000_w0.01_suf_smooth_100_lrpose0.0005xobj1e-05_exp/meshes/00007501_obj.ply')
    parser.add_argument('--target', type=str, default='/home/yufeiy2/scratch/result/HOI4D/Mug_1/oObj.obj')
    parser.add_argument('--flip_x', action='store_true')
    parser.add_argument('--flip_y', action='store_true')
    parser.add_argument('--flip_z', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()

    if args.test:
        test()
    else:
        compare(args.source, args.target, args.iter, args.flip_x, args.flip_y, args.flip_z)
