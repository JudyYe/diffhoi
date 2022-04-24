import numpy as np
import os
import os.path as osp

from tqdm import tqdm
from models.frameworks.volsdf_hoi import MeshRenderer, VolSDFHoi
from preprocess.smooth_hand import vis_hA, vis_se3
from utils import io_util, mesh_util
from utils.dist_util import is_master

from models.frameworks import get_model
from models.cameras import get_camera
from dataio import get_data

import torch
from torch.utils.data.dataloader import DataLoader
from jutils import image_utils, geom_utils, mesh_utils


def run_render(dataloader:DataLoader, trainer:VolSDFHoi, save_dir, name, render_kwargs, offset=None):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    os.makedirs(save_dir, exist_ok=True)

    orig_H, orig_W = dataloader.dataset.H, dataloader.dataset.W

    # if H is not None:
    #     render_kwargs['H'] = H
    # if W is not None:
    #     render_kwargs['W'] = W
    H, W = render_kwargs['H'], render_kwargs['W']
    
    # reconstruct  hand and render
    name_list = ['gt', 'render_0', 'render_1', 'hand_0', 'hand_1', 'obj_0', 'obj_1', 'hand_front', 'obj_front']
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
        image_list[3].append(image1['hand'])
        image_list[4].append(image2['hand'])
        image_list[5].append(image1['obj'])
        image_list[6].append(image2['obj'])
        image_list[7].append(image1['hand_front'])
        image_list[8].append(image1['obj_front'])

    for n, img_list in zip(name_list, image_list):
        image_utils.save_gif(img_list, osp.join(save_dir, name + '_%s' % n))

    file_list = [osp.join(save_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list



def run_hA(dataloader, trainer, save_dir, name):
    device = trainer.device
    name_list = ['hHand', 'hHand_1']
    image_list = [[] for _ in name_list]

    hA_list, jTc_list, jTh_list = [], [], []
    for (indices, model_input, ground_truth) in dataloader:        
        for k, v in model_input.items():
            try:
                model_input[k] = v.to(device)
            except AttributeError:
                pass
        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, 200, 200)
        hHand = mesh_utils.apply_transform(jHand, geom_utils.inverse_rt(mat=jTh, return_mat=True))
        hA = trainer.model.hA_net(indices.to(device), model_input, None)
        # gif = mesh_utils.render_geom_rot(hHand, time_len=3, scale_geom=True)
        
        hA_list.append(hA[0].cpu())
        jTc_list.append(jTc[0].cpu().detach().numpy())
        jTh_list.append(jTh[0].cpu().detach().numpy())

        # image_list[0].append(gif[0])
        # image_list[1].append(gif[1])

    vis_se3(jTc_list, osp.join(save_dir, name + '_jTc'), 'jTc')
    vis_se3(jTh_list, osp.join(save_dir, name + '_jTh'), 'jTh')
    vis_hA(hA_list, osp.join(save_dir, name + '_hA'), 'hA', range(6))

    # for n, img_list in zip(name_list, image_list):
        # image_utils.save_gif(img_list, osp.join(save_dir, name + '_%s' % n))

    file_list = [osp.join(save_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list

def run_gt(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6):
    device = trainer.device

    renderer = trainer.mesh_renderer

    os.makedirs(save_dir, exist_ok=True)
    # reconstruct  hand and render in normazlied frame
    name_list = ['rgb', 'mask', 'sem_masks']
    image_list = [[] for _ in name_list]
    for (indices, model_input, ground_truth) in dataloader:
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        hand_mask = model_input['hand_mask'].reshape(1, hh, ww, 1).permute(0, 3, 1, 2)
        obj_mask = model_input['obj_mask'].reshape(1, hh, ww, 1).permute(0, 3, 1, 2)

        sem_mask = torch.cat([hand_mask, obj_mask, torch.zeros_like(obj_mask)], 1)
        mask = (hand_mask + obj_mask).clamp(max=1)

        image_list[0].append(gt)
        image_list[1].append(image_utils.blend_images(sem_mask, gt, mask))  # view 0
        image_list[2].append(sem_mask)  # view 1
        # toto: render novel view!
    if is_master():
        for n, im_list in zip(name_list, image_list):
            image_utils.save_gif(im_list, osp.join(save_dir, name + '_%s' % n))

    file_list = [osp.join(save_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list


def run(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 2]]).to(device), 
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
    name_list = ['gt', 'view_0', 'view_1', 'view_j', 'view_h', 'view_hy', 'obj']
    image_list = [[] for _ in name_list]
    for (indices, model_input, ground_truth) in dataloader:
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        mesh_utils.dump_meshes([osp.join(save_dir, name + '_jHand', '%d' % indices[0].item())], jHand)
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
        image_list[1].append(image_utils.blend_images(image1, gt, mask1))  # view 0
        image_list[2].append(image2)  # view 1
        image_list[3].append(image3)  # view_j 
        image_list[4].append(image4)  # view _h 
        image_list[5].append(image5)

    image_list[6] = mesh_utils.render_geom_rot(jObj, scale_geom=True)
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
    model, trainer, render_kwargs_train, render_kwargs_test, _, _ = get_model(config, data_size=len(dataset)+1, cam_norm=dataset.max_cam_norm, device=[0])

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

    if args.gt:
        with torch.no_grad():
            run_gt(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N)
    if args.surface:
        with torch.no_grad():
            run(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N)
    if args.nvs:
        render_kwargs_test['rayschunk'] = args.chunk
        render_kwargs_test['H'] = H  * 2 // 3
        render_kwargs_test['W'] = W  * 2 // 3
        with torch.no_grad():
            run_render(dataloader, trainer, save_dir, name, render_kwargs_test,)
    if args.hand:
        with torch.no_grad():
            save_dir = args.load_pt.split('/ckpts')[0] + '/hA/'
            run_hA(dataloader, trainer, save_dir, name, )


def render(renderer, jHand, jObj, jTc, intrinsics, H, W, zfar=-1):
    jHand.textures = mesh_utils.pad_texture(jHand, 'blue')
    jObj.textures = mesh_utils.pad_texture(jObj, 'white')
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
    return image.cpu(), mask.cpu()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help='output ply file name')
    parser.add_argument('--N', type=int, default=64, help='resolution of the marching cube algo')
    parser.add_argument('--volume_size', type=float, default=6., help='voxel size to run marching cube')
    parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    parser.add_argument("--chunk", type=int, default=256, help='net chunk when querying the network. change for smaller GPU memory.')
    parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    parser.add_argument("--hand", action='store_true')
    parser.add_argument("--nvs", action='store_true')
    parser.add_argument("--gt", action='store_true')
    parser.add_argument("--surface", action='store_true')
    args = parser.parse_args()
    
    main_function(args)