from fileinput import filelineno
from flask import render_template
import numpy as np
import os
import os.path as osp

from tqdm import tqdm
from models.frameworks.volsdf_hoi import MeshRenderer
from utils import io_util, mesh_util
from utils.dist_util import is_master
from utils.mesh_util import extract_mesh

from models.frameworks import get_model
from models.cameras import get_camera
from dataio import get_data

import torch
from torch.utils.data.dataloader import DataLoader
from jutils import image_utils, geom_utils, mesh_utils


def run_render(dataloader:DataLoader, trainer, save_dir, name, render_kwargs, H=None, W=None, offset=None):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    os.makedirs(save_dir, exist_ok=True)

    orig_H, orig_W = dataloader.dataset.H, dataloader.dataset.W

    if H is not None:
        render_kwargs['H'] = H
    if W is not None:
        render_kwargs['W'] = W
    H, W = render_kwargs['H'], render_kwargs['W']
    
    # reconstruct  hand and render
    name_list = ['gt', 'render_0', 'render_1', 'hand_front', 'obj_front']
    image_list = [[] for _ in name_list]
    for (indices, model_input, ground_truth) in tqdm(dataloader):
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        image_list[0].append(gt)

        jHand, jTc, _, intrinsics = trainer.get_jHand_camera(indices.to(device), model_input, ground_truth, H, W)

        intrinsics[..., 0, 2] /= orig_W / W 
        intrinsics[..., 0, 0] /= orig_W / W 
        intrinsics[..., 1, 2] /= orig_H / H 
        intrinsics[..., 1, 1] /= orig_H / H 

        with torch.no_grad():
            image1 = trainer.render(jHand, jTc, intrinsics, render_kwargs)

            cam_norm = mesh_utils.get_camera_dist(wTc=jTc)
            xyz = torch.cat([torch.zeros_like(cam_norm), torch.zeros_like(cam_norm), -cam_norm * 0.1])
            z_back = geom_utils.axis_angle_t_to_matrix(t=xyz)
            image2 = trainer.render(jHand, jTc@z_back@offset, intrinsics, render_kwargs)

        image_list[1].append(image1['image'])
        image_list[2].append(image2['image'])
        image_list[3].append(image1['hand_front'])
        image_list[4].append(image1['obj_front'])

    for n, img_list in zip(name_list, image_list):
        image_utils.save_gif(img_list, osp.join(save_dir, name + '_%s' % n))


def run(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    rot_y = geom_utils.axis_angle_t_to_matrix(
        np.pi / 2 * torch.FloatTensor([[1, 0, 0]]).to(device), 
    )

    model = trainer.model
    renderer = trainer.mesh_renderer

    os.makedirs(save_dir, exist_ok=True)
    # reconstruct object
    if is_master():
        mesh_util.extract_mesh(
            model.implicit_surface, 
            N=N,
            filepath=osp.join(save_dir, name + '_obj.ply'),
            volume_size=volume_size,
        )
        jObj = mesh_utils.load_mesh(osp.join(save_dir, name + '_obj.ply')).cuda()

    # reconstruct  hand and render in normazlied frame
    name_list = ['gt', 'viwe_0', 'view_1', 'view_j', 'view_h', 'view_hy']
    image_list = [[] for _ in name_list]
    for (indices, model_input, ground_truth) in dataloader:
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        hTj = geom_utils.inverse_rt(mat=jTh, return_mat=True)
        
        image1, mask1 = render(renderer, jHand, jObj, jTc, intrinsics, H, W)
        
        cam_norm = mesh_utils.get_camera_dist(wTc=jTc)
        xyz = torch.cat([torch.zeros_like(cam_norm), torch.zeros_like(cam_norm), -cam_norm * 0.1])
        z_back = geom_utils.axis_angle_t_to_matrix(t=xyz)

        image2, _ = render(renderer, jHand, jObj, jTc@z_back@offset, intrinsics, H, W)
        image3, _ = render(renderer, jHand, jObj, None, None, H, W)
        image4, _ = render(renderer, 
            mesh_utils.apply_transform(jHand, hTj), 
            mesh_utils.apply_transform(jObj, hTj), None, None, H, W)
        image5, _ = render(renderer, 
            mesh_utils.apply_transform(jHand,  rot_y@hTj), 
            mesh_utils.apply_transform(jObj, rot_y@hTj), None, None, H, W)

        image_list[0].append(gt)
        image_list[1].append(image_utils.blend_images(image1, gt, mask1))
        image_list[2].append(image2)
        image_list[3].append(image3)
        image_list[4].append(image4)
        image_list[5].append(image5)

        # toto: render novel view!
    if is_master():
        for n, im_list in zip(name_list, image_list):
            image_utils.save_gif(im_list, osp.join(save_dir, name + '_%s' % n))

    file_list = [osp.join(save_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list


def main_function(args):

    H = W = 224
    device = 'cuda:0'
    # load config
    config = io_util.load_yaml(osp.join(args.load_pt.split('/ckpts')[0], 'config.yaml'))

    # load data    
    dataset, _ = get_data(config, return_val=True, val_downscale=1)
    dataloader = DataLoader(dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mesh_utils.collate_meshes)

    # build and load model 
    posenet, focal_net = get_camera(config, datasize=len(dataset)+1, H=dataset.H, W=dataset.W)
    model, trainer, render_kwargs_train, render_kwargs_test, _, _ = get_model(config, data_size=len(dataset)+1, cam_norm=dataset.max_cam_norm)

    assert args.out is not None or args.load_pt is not None, 'Need to specify one of out / load_pt'
    
    if args.load_pt is not None:
        state_dict = torch.load(args.load_pt)
        
        model.load_state_dict(state_dict['model'])
        posenet.load_state_dict(state_dict['posenet'])
        focal_net.load_state_dict(state_dict['focalnet'])
        
        trainer.init_camera(posenet, focal_net)
        trainer.to(device)
        trainer.eval()
        
        it = state_dict['global_step']
        name = 'it%08d' % it
        save_dir = args.load_pt.split('/ckpts')[0] + '/render/'
    
    if args.out is not None:
        save_dir = args.out

    os.makedirs(save_dir, exist_ok=True)

    # run(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size)
    render_kwargs_test['rayschunk'] = args.chunk
    render_kwargs_test['H'] = H  * 2 // 3
    render_kwargs_test['W'] = W  * 2 // 3
    run_render(dataloader, trainer, save_dir, name, render_kwargs_test,)


def render(renderer, jHand, jObj, jTc, intrinsics, H, W, zfar=-1):
    jMeshes = mesh_utils.join_scene([jHand, jObj])
    if jTc is None:
        image = mesh_utils.render_geom_rot(jMeshes, scale_geom=True, time_len=1, out_size=H)[0]
        mask = torch.ones_like(image)
    else:
        iMesh = renderer(
            geom_utils.inverse_rt(mat=jTc, return_mat=True), 
            intrinsics, 
            jMeshes, 
            H, W, zfar,
            )
        image = iMesh['image']
        mask = iMesh['mask']
    return image, mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help='output ply file name')
    parser.add_argument('--N', type=int, default=64, help='resolution of the marching cube algo')
    parser.add_argument('--volume_size', type=float, default=6., help='voxel size to run marching cube')
    parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    parser.add_argument("--chunk", type=int, default=1024, help='net chunk when querying the network. change for smaller GPU memory.')
    parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    args = parser.parse_args()
    
    main_function(args)