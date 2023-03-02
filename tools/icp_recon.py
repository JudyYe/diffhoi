
import os
import os.path as osp
import trimesh
import numpy as np
import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from chamferdist import ChamferDistance
from pytorch3d.transforms import Transform3d
from pytorch3d.loss import point_mesh_distance, chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from jutils import image_utils, mesh_utils, geom_utils
from jutils.mesh_utils import Meshes
from jutils.geom_utils import Rotate


def register_meshes(source: Meshes, target: Meshes, scale=True, N=100, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    # first normalize bc icp only works with mesh with coarse alignment
    target, cTo_t = mesh_utils.center_norm_geom(target, )
    cSource = mesh_utils.apply_transform(source, cTo_t)
    source, cTo_s = mesh_utils.center_norm_geom(cSource, max_norm=None)

    new_source, new_target, targetTsource = my_register(source, target, N, scale)

    new_target = mesh_utils.apply_transform(new_target, cTo_t.inverse())
    new_source = mesh_utils.apply_transform(new_source, cTo_t.inverse())
    
    return new_source, new_target


def my_register(source:Meshes, target: Meshes, N, allow_scale, T=10):
    # SGD to optimize the transformation s,R,t
    BS = len(source)
    device = source.device
    def init_param():
        scale = torch.ones([BS*N, 1], device=device)
        rot =  geom_utils.matrix_to_rotation_6d(
            geom_utils.random_rotations(BS*N, device=device))
        trans = torch.zeros([BS*N, 3], device=device)
        return scale, rot, trans
    scale, rot, trans = init_param()
    if allow_scale:
        scale = nn.Parameter(scale)
    rot = nn.Parameter(rot)
    trans = nn.Parameter(trans)
    para_list = [rot, trans]
    if allow_scale:
        para_list.append(scale)
    
    optimizer = torch.optim.LBFGS(para_list, lr=0.1, max_iter=100, max_eval=100, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn='strong_wolfe')
    # duplicate mesh (BS*N? or N*BS??)
    dup_source = Meshes(source.verts_padded().repeat(N, 1, 1), source.faces_padded().repeat(N, 1, 1)).to(device)  # B*N
    dup_target = Meshes(target.verts_padded().repeat(N, 1, 1), target.faces_padded().repeat(N, 1, 1)).to(device)

    P = 1000
    points_source = sample_points_from_meshes(dup_source, P)
    points_target = sample_points_from_meshes(dup_target, P)

    def get_points():
        targetTsource = geom_utils.rt_to_homo(
            geom_utils.rotation_6d_to_matrix(rot), trans, scale)
        # if find NaN, reset rot, trans, scale
        nan_mask = torch.isnan(targetTsource).sum(dim=(1,2)) > 0
        if nan_mask.any():
            print('nan????')
            new_scale, new_rot, new_trans = init_param()
            scale.data[nan_mask] = new_scale[nan_mask]
            rot.data[nan_mask] = new_rot[nan_mask]
            trans.data[nan_mask] = new_trans[nan_mask]
            targetTsource = geom_utils.rt_to_homo(
                geom_utils.rotation_6d_to_matrix(rot), trans, scale)

        s_points = Transform3d(matrix=targetTsource.transpose(-1, -2)).transform_points(points_source)
        t_points = points_target
        return s_points, t_points        
    
    # SGD on the transformation that minimize the distance
    def closure():
        optimizer.zero_grad()
        s_points, t_points = get_points()
        # compute 2 way chamfer distance?
        dist, _ = chamfer_distance(s_points, t_points, )
        dist *= 1000
        dist.backward()
        return dist

    for i in range(T):
        optimizer.step(closure)

    # get the best transformation among N 
    s_points, t_points = get_points()
    dist, _  = chamfer_distance(s_points, t_points, batch_reduction=None)
    dist = dist.reshape([BS, N])
    ind = torch.argmin(dist, dim=1) # (BS,) in [0, N)
    targetTsource = geom_utils.rt_to_homo(
        geom_utils.rotation_6d_to_matrix(rot), trans, scale)
    targetTsource = targetTsource.reshape([BS, N, 4, 4])
    best_targetTsource = torch.gather(targetTsource, 1, ind.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, 4, 4))
    best_targetTsource = best_targetTsource.squeeze(1)
    new_source = mesh_utils.apply_transform(source, best_targetTsource)
    return new_source, target, best_targetTsource

    

def register_meshes_legacy(source: Meshes, target: Meshes, type='icp_common', scale=True, N=1):
    device = source.device
    BS = len(source)
    cost_list = np.zeros([BS, N]) + 100000000
    def register_one(source, target, n):
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
        for i, (s, t) in enumerate(zip(source_list, target_list)):
            s, t, cost = register(s, t, type, scale=scale, samples=5000, icp_first=100, icp_final=500)
            new_source_list.append(s)
            new_target_list.append(t)
            cost_list[i, n] = cost

        new_source = mesh_utils.from_trimeshes(new_source_list).to(device)
        new_target = mesh_utils.from_trimeshes(new_target_list).to(device)

        new_target = mesh_utils.apply_transform(new_target, cTo_t.inverse())
        new_source = mesh_utils.apply_transform(new_source, cTo_t.inverse())
        return new_source, new_target

    rots = geom_utils.random_rotations(N, device=device)
    rots[0] = torch.eye(3, device=device)
    record_list = []
    for n in tqdm(range(N), desc='randomize rotation'):
        rot = Rotate(rots[n:n+1].repeat(BS, 1, 1), device=device)
        rot_source = mesh_utils.apply_transform(source, rot)
        new_source, new_target = register_one(rot_source, target, n)
        record_list.append((new_source, new_target, ))
    idx = np.argmin(cost_list, axis=-1)  # bs

    # select the best one in 
    new_source_list = []
    new_target_list = []
    for i in range(BS):
        new_source, new_target = record_list[idx[i]]
        new_source_list.append(new_source.verts_padded()[i])
        new_target_list.append(new_target.verts_padded()[i])

    new_source_verts = torch.stack(new_source_list, dim=0)
    new_target_verts = torch.stack(new_target_list, dim=0)
    new_source = source.clone()
    new_target = target.clone()
    new_source = new_source.update_padded(new_source_verts)
    new_target = new_target.update_padded(new_target_verts)
    return new_source, new_target


def register(source, target, type='icp_common', scale=True, **kwargs):
    if type == 'icp_common':
        from trimesh.registration import mesh_other
    elif type == 'icp_constrained':
        from tools.icp import mesh_other
    else:
        raise ValueError('Registration Type Should Be in {icp_common} and {icp_constrained}.')

    # register
    source2target, cost = mesh_other(source, target, scale=scale, **kwargs)
    # source2target, cost = mesh_other(source, target, scale=False)

    # transform
    source.apply_transform(source2target)

    return source, target, cost


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
        new_source, _, _ = register(source, target)
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
    torch.manual_seed(123)
    data_dir = '/home/yufeiy2/scratch/result/vis'
    device = 'cuda:0'

    mesh_s = mesh_utils.load_mesh(args.source, device=device, scale_verts=0.1)
    mesh_f = mesh_utils.load_mesh(args.target, device=device)
    hoi = mesh_utils.join_scene_w_labels([mesh_s, mesh_f])
    image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(data_dir, 'before'))

    N = args.iter
    # new_s, _ = register_meshes(mesh_s, mesh_f, N=N, scale=scale)
    new_s, _ = register_meshes(mesh_s, mesh_f, N=N, scale=scale)

    hoi = mesh_utils.join_scene_w_labels([new_s, mesh_f])
    image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(data_dir, 'align_N%d_scale%d' % (N, scale)))
    mesh_utils.dump_meshes([osp.join(data_dir, 'align_N%d_scale%d'%(N, scale))], new_s)
    mesh_utils.dump_meshes([osp.join(data_dir, 'before')], mesh_f)

    th_list = np.array([10, 20]) * 1e-3
    f_list = mesh_utils.fscore(new_s, mesh_f, th=th_list)

    print(f_list)

    # for i in tqdm(range(args.iter)):
    #     # register
    #     rot = geom_utils.random_rotations(1, device=device)
    #     rot = Rotate(rot, device=device)
    #     new_s, _ = register_meshes(mesh_utils.apply_transform(mesh_s, rot), mesh_f)

        # hoi = mesh_utils.join_scene_w_labels([new_s, mesh_f])
        # image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
        # image_utils.save_gif(image_list, osp.join(data_dir, 'align_%d' % i))
        # mesh_utils.dump_meshes([osp.join(data_dir, 'align_%d' % i)], mesh_s)

        # th_list = np.array([10]) * 1e-3
        # f_list = mesh_utils.fscore(new_s, mesh_f, th=th_list)

        # print(f_list)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/yufeiy2/scratch/result/vhoi/which_prior/Bowl_1_suf_smooth_100_CondGeomGlide_cond_all_linear_catTrue_cfgFalse/meshes/00045001.ply')
    parser.add_argument('--target', type=str, default='/home/yufeiy2/scratch/result/HOI4D/Bowl_1/oObj.obj')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--flip_x', action='store_true')
    parser.add_argument('--flip_y', action='store_true')
    parser.add_argument('--flip_z', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()

    scale = args.scale
    if args.test:
        test()
    else:
        compare(args.source, args.target, args.iter, args.flip_x, args.flip_y, args.flip_z)
