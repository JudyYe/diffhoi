
import os
import os.path as osp
import trimesh
import numpy as np
import copy
import torch
from tqdm import tqdm

from chamferdist import ChamferDistance

from jutils import image_utils, mesh_utils, geom_utils
from jutils.mesh_utils import Meshes

def register(source, target, type='icp_common'):
    if type == 'icp_common':
        from trimesh.registration import mesh_other
    elif type == 'icp_constrained':
        from tools.icp import mesh_other
    else:
        raise ValueError('Registration Type Should Be in {icp_common} and {icp_constrained}.')

    # register
    source2target, cost = mesh_other(source, target, scale=True)
    # source2target, cost = mesh_other(source, target, scale=False)

    # transform
    source.apply_transform(source2target)

    return source, target


def compare(source_file, target_file, iters=1, flip_x=False, flip_y=False, flip_z=False):
    chamferDist = ChamferDistance()

    source = trimesh.load(source_file)  # reconstructed mesh
    target = trimesh.load(target_file)  # ground truth mesh

    # flip
    if flip_x:
        source.apply_translation([-source.centroid[0], 0, 0])
    if flip_y:
        source.apply_translation([0, -source.centroid[1], 0])
    if flip_z:
        source.apply_translation([0, 0, -source.centroid[2]])

    # normalize
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
    
    return dist_bidirectional

def test():
    data_dir = '/home/yufeiy2/scratch/result/vis'
    device = 'cuda:0'
    source_file = osp.join(data_dir, 'a.ply')
    target_file = osp.join(data_dir, 'b.obj')
    
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
    mesh_utils.dump_meshes([osp.join(data_dir, 'before_s')], mesh_s)
    mesh_utils.dump_meshes([osp.join(data_dir, 'before_f')], mesh_f)
    for i in range(args.iter):

        # register
        new_source, _ = register(source, target)
    
        mesh_s = Meshes(
            [torch.FloatTensor(new_source.vertices)], 
            [torch.LongTensor(new_source.faces)]).to(device)
        hoi = mesh_utils.join_scene_w_labels([mesh_s, mesh_f])
        image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(data_dir, 'align_%d' % i))
        mesh_utils.dump_meshes([osp.join(data_dir, 'align_%d' % i)], mesh_s)
        # new_source = source



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/dihuang/data/mesh/source.ply')
    parser.add_argument('--target', type=str, default='/home/dihuang/data/mesh/target.ply')
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
