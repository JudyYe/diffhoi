# ICP and chamfer distance evaluation for HHOR meshes
import json
import numpy as np
import trimesh
import torch
from tqdm import tqdm
import os.path as osp
import os
from glob import glob
import argparse
from tools.icp_recon import compare
import matplotlib.pyplot as plt
from jutils import image_utils, mesh_utils, geom_utils 
from pytorch3d.structures import Meshes
device = 'cuda:0'


def compare_meshes(source, target):
    """_summary_

    :param source: _description_
    :param target: _description_
    :return: _description_
    """

    return cd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/yufeiy2/scratch/result/hhor/understand/barfTrue_warpTrue_warmup5000/handscanner/meshes/')
    parser.add_argument('--target', type=str, default='/home/yufeiy2/scratch/data/HHOR/CAD/')
    parser.add_argument('--index', type=str, default='Sculptures/3_Giuliano')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--flip_x', action='store_true')
    parser.add_argument('--flip_y', action='store_true')
    parser.add_argument('--flip_z', action='store_true')
    args = parser.parse_args()
    return args


def vis_alignment(source, target, name, save_dir):
    def make_mesh(mesh):
        vertices, faces = mesh
        mesh = Meshes([vertices[0]], [torch.LongTensor(faces).to(vertices)])
        return mesh
    
    source = make_mesh(source).to(device)
    target = make_mesh(target).to(device)
    hoi = mesh_utils.join_scene_w_labels([source, target])
    image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, name))
    # render source and target separately
    image_list = mesh_utils.render_geom_rot(source, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, name + '_source'))
    image_list = mesh_utils.render_geom_rot(target, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, name + '_target'))


def postprocess_mesh(mesh, num_faces=None):
    """Post processing mesh by removing small isolated pieces.

    Args:
        mesh (trimesh.Trimesh): input mesh to be processed
        num_faces (int, optional): min face num threshold. Defaults to 4096.
    """
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 100
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency)
    mask = np.zeros(total_num_faces, dtype=np.bool)
    for i, c in enumerate(cc):
        print(i, len(c))
    
    # find maximum connected component cc and return the index
    max_cc = np.argmax([len(c) for c in cc])

    # cc = np.concatenate([
    #     c for c in cc if len(c) > num_faces
    # ], axis=0)
    cc = cc[max_cc]
    mask[cc] = True
    print('before', mesh.vertices.shape, mesh.faces.shape)
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    print('after', mesh.vertices.shape, mesh.faces.shape)
    return mesh



def main():
    save_dir = osp.join(args.source, '../metrics')
    os.makedirs(save_dir, exist_ok=True)

    cd_list = []
    x_list = []
    mesh_list = sorted(glob(osp.join(args.source, '*_obj.ply')), reverse=True)
    for mesh_file in tqdm(mesh_list[0::5]):
        t = int(mesh_file.split('/')[-1].split('_')[0])
        target = osp.join(args.target, args.index + '.ply')
        mesh = trimesh.load(mesh_file)
        mesh = postprocess_mesh(mesh)
        cd, s_mesh, t_mesh = compare(mesh, target, )
        cd = cd.detach().cpu().item()
        # vis_alignment(s_mesh, t_mesh, osp.basename(mesh_file), save_dir)
        print(cd)

        x_list.append(t)
        cd_list.append(cd)
    
    plt.plot(x_list, cd_list)
    plt.xlabel('Iteration')
    plt.ylabel('Chamfer Distance')
    plt.savefig(osp.join(save_dir, 'cd.png'))

    json_file = osp.join(save_dir, 'cd.json')
    with open(json_file, 'w') as f:
        json.dump({'x': x_list, 'y': cd_list}, f)
                     
    return 

if __name__ == '__main__':
    args = parse_args()
    main()
