from glob import glob
import hydra
from functools import partial
import numpy as np
import os
import os.path as osp
from time import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from models.frameworks.volsdf_hoi import VolSDFHoi, Trainer
# from preprocess.smooth_hand import vis_hA, vis_se3
from utils import io_util, mesh_util
from utils.dist_util import is_master

from models.frameworks import get_model
from models.cameras import get_camera
from dataio import get_data

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from jutils import image_utils, geom_utils, mesh_utils, model_utils, plot_utils, slurm_utils

def run_train_render(dataloader:DataLoader, trainer:Trainer, save_dir, name, render_kwargs, offset=None):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    os.makedirs(save_dir, exist_ok=True)

    orig_H, orig_W = dataloader.dataset.H, dataloader.dataset.W
    H, W = render_kwargs['H'], render_kwargs['W']
    
    # reconstruct  hand and render
    name_list = ['gt', 'render_0', 'render_1', 'hand_0', 'hand_1', 'obj_0', 'obj_1', 'hand_front', 'obj_front']
    name_list = ['train_' + e for e in name_list]
    image_list = [[] for _ in name_list]
    for (indices, model_input, ground_truth) in tqdm(dataloader):
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        image_list[0].append(gt)

        jHand, jTc, _, intrinsics = trainer.get_jHand_camera(indices.to(device), model_input, ground_truth, H, W)

        trainer.zero_grad()
        with torch.no_grad():
            t = time()
            image1 = trainer.render(jHand, jTc, intrinsics, render_kwargs, use_surface_render=None)
            duration = time() - t
            mem_usage = torch.cuda.memory_allocated(0)/1024/1024/1024
            mem_total = 24 # torch.cuda.max(0)/1024/1024/1024
            print('resolution:', image1['image'].shape)
            print(f'render one image time:{duration:.2f} s')
            print(f'mem: {mem_usage:.2f}GB / {mem_total:.2f}GB = {mem_usage / mem_total * 100:.2f}\%')

            cam_norm = mesh_utils.get_camera_dist(wTc=jTc)
            xyz = torch.cat([torch.zeros_like(cam_norm), torch.zeros_like(cam_norm), -cam_norm * 0.1])
            z_back = geom_utils.axis_angle_t_to_matrix(t=xyz)
            # image2 = trainer.render(jHand, jTc@z_back@offset, intrinsics, render_kwargs)

        image_list[1].append(image1['image'])
        # image_list[2].append(image2['image'])
        image_list[3].append(image1['hand'])
        # image_list[4].append(image2['hand'])
        image_list[5].append(image1['obj'])
        # image_list[6].append(image2['obj'])
        image_list[7].append(image1['hand_front'])
        image_list[8].append(image1['obj_front'])
        break

    for n, img_list in zip(name_list, image_list):
        image_utils.save_gif(img_list, osp.join(save_dir, name + '_%s' % n))

    file_list = [osp.join(save_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list


def run_vis_diffusion(dataloader:DataLoader, trainer:Trainer, save_dir, name, render_kwargs, offset=None):
    device = trainer.device
    if isinstance(trainer, DistributedDataParallel):
        trainer = trainer.module
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    trainer.train()
    for (indices, model_input, ground_truth) in tqdm(dataloader):
        cnt += 1
        indices = indices.to(device)
        model_utils.to_cuda(model_input, device)
        model_utils.to_cuda(ground_truth, device)
        # print(model_input['intrinsics'])
        for i in range(10):
            extras = trainer(trainer.args, indices, model_input, ground_truth, render_kwargs, 0, False)
            iHand = extras['iHand']

            img = trainer.get_diffusion_image(extras)
            image_dict = {}
            trainer.sd_loss.model.vis_samples({}, img, None, '%d_' % cnt, image_dict)
            # save 
            for k, v in image_dict.items():
                save_path = osp.join(save_dir, name + '_' + k.replace('/', '_') + f'{i}.png')
                v.image.save(save_path)


@torch.no_grad()
def run_video(dataloader, trainer, save_dir, name, H, W,  N=64, volume_size=6, max_t=1000, T_num=None):
    device = trainer.device
    if isinstance(trainer, DistributedDataParallel):
        trainer = trainer.module
    model = trainer.model
    save_dir = osp.join(save_dir, name, )
    os.makedirs(save_dir, exist_ok=True)
    mesh_file = osp.join(save_dir, 'jObj.ply')
    if not osp.exists(mesh_file):
        try:
            mesh_util.extract_mesh(
                    model.implicit_surface, 
                    N=N,
                    filepath=mesh_file,
                    volume_size=volume_size,
                )
        except ValueError:
            jObj = plot_utils.create_coord('cpu', size=0.00001)
            mesh_utils.dump_meshes([mesh_file[:-4]], jObj)
            mesh_file = mesh_file.replace('.ply', '.obj')
    jObj = mesh_utils.load_mesh(mesh_file).cuda()
    jObj.textures = mesh_utils.pad_texture(jObj, 'blue')

    name_list = ['input', 'render_0', 'render_1', 'jHoi', 'jObj', 'vHoi', 'vObj', 'vObj_t', 'vHoi_fix']
    image_list = [[] for _ in name_list]
    T = len(dataloader)
    for i, (indices, model_input, ground_truth) in enumerate(tqdm(dataloader)):
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        image_list[0].append(gt)

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        jHand.textures = mesh_utils.pad_texture(jHand, 'blue')

        K_ndc = mesh_utils.intr_from_screen_to_ndc(intrinsics, H, W)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTc, H=H, K_ndc=K_ndc, bin_size=32)
        image_list[1].append(hoi)

        # rotate by 90 degree in world frame 
        # 1. 
        jTcp = mesh_utils.get_wTcp_in_camera_view(np.pi/2, wTc=jTc)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTcp, H=H, K_ndc=K_ndc, bin_size=32)
        image_list[2].append(hoi)

        if i == T//2:
            # coord = plot_utils.create_coord(device, size=1)
            jHoi = mesh_utils.join_scene([jHand, jObj])
            image_list[3] = mesh_utils.render_geom_rot(jHoi, scale_geom=True, out_size=H) 
            image_list[4] = mesh_utils.render_geom_rot(jObj, scale_geom=True, out_size=H) 
            
            # rotation around z axis
            vTj = torch.FloatTensor(
                [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
                [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
            vHoi = mesh_utils.apply_transform(jHoi, vTj)
            vObj = mesh_utils.apply_transform(jObj, vTj)
            image_list[5] = mesh_utils.render_geom_rot(vHoi, scale_geom=True, out_size=H) 
            image_list[6] = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H) 
        
        jHoi = mesh_utils.join_scene([jHand, jObj])                
        vTj = torch.FloatTensor(
            [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
            [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
        vObj = mesh_utils.apply_transform(jObj, vTj)
        iObj_list = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H, bin_size=32) 
        image_list[7].append(iObj_list[i%len(iObj_list)])
        
        # HOI from fixed view point 
        scale = mesh_utils.Scale(0.5, ).cuda()
        trans = mesh_utils.Translate(0, 0.4, 0, ).cuda()
        fTj = scale.compose(trans)
        fHand = mesh_utils.apply_transform(jHand, fTj)
        fObj = mesh_utils.apply_transform(jObj, fTj)
        iHoi, iObj = mesh_utils.render_hoi_obj(fHand, fObj, 0, scale_geom=False, scale=1, bin_size=32)
        image_list[8].append(iHoi)
    # save 
    for n, im_list in zip(name_list, image_list):
        for t, im in enumerate(im_list):
            image_utils.save_images(im, osp.join(save_dir, n, f'{t:03d}'))
    return

def run_render(dataloader:DataLoader, trainer:VolSDFHoi, save_dir, name, render_kwargs, offset=None, max_t=1000):
    device = trainer.device
    if isinstance(trainer, DistributedDataParallel):
        trainer = trainer.module
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    os.makedirs(save_dir, exist_ok=True)

    H, W = render_kwargs['H'], render_kwargs['W']
    
    # reconstruct  hand and render
    name_list = ['gt', 'render_0', 'render_1', 'hand_0', 'hand_1', 'obj_0', 'obj_1', 'hand_front', 'obj_front']
    image_list = [[] for _ in name_list]
    for i, (indices, model_input, ground_truth) in enumerate(tqdm(dataloader)):
        if i >= max_t:
            break
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        image_list[0].append(gt)

        jHand, jTc, _, intrinsics = trainer.get_jHand_camera(indices.to(device), model_input, ground_truth, H, W)

        with torch.no_grad():
            t = time()
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


def run_vis_cam(dataloader:DataLoader, trainer:Trainer, save_dir, name, render_kwargs):
    device = trainer.device
    if isinstance(trainer, DistributedDataParallel):
        trainer = trainer.module
    os.makedirs(save_dir, exist_ok=True)

    H, W = render_kwargs['H'], render_kwargs['W']
    jTc_list, jTc_nv_list = [], []
    for (indices, model_input, ground_truth) in tqdm(dataloader):
        indices = indices.to(device)
        model_utils.to_cuda(model_input)
        model_utils.to_cuda(ground_truth)
        jHand, jTc, _, intrinsics = trainer.get_jHand_camera(indices, model_input, ground_truth, H, W)
        jTc_nv =trainer.sample_jTc(indices, model_input, ground_truth)[0]
        jTc_list.append(jTc)
        jTc_nv_list.append(jTc_nv)
    
    jTc = torch.cat(jTc_list)
    jTc_nv = torch.cat(jTc_nv_list)
    N = len(jTc)
    coord = plot_utils.create_coord(device, N, 0.1)

    print(
        geom_utils.mat_to_scale_rot(jTc[..., :3, :3])[1], 
        geom_utils.mat_to_scale_rot(jTc_nv[..., :3, :3])[1])
    wScene_origin = plot_utils.vis_cam(wTc=jTc, color='red', size=0.02)
    # print(wScene_origin.)
    scene = mesh_utils.join_scene(wScene_origin + [coord,jHand])
    image_list = mesh_utils.render_geom_rot(scene, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, name + '_%s' % 'origin'))

    wScene_novel = plot_utils.vis_cam(wTc=jTc_nv, color='blue', size=0.02)
    scene = mesh_utils.join_scene(wScene_origin + wScene_novel +  [coord, jHand])
    image_list = mesh_utils.render_geom_rot(scene, scale_geom=True, out_size=1024)
    image_utils.save_gif(image_list, osp.join(save_dir, name + '_%s' % 'novel'))
    

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

    # vis_se3(jTc_list, osp.join(save_dir, name + '_jTc'), 'jTc')
    # vis_se3(jTh_list, osp.join(save_dir, name + '_jTh'), 'jTh')
    # vis_hA(hA_list, osp.join(save_dir, name + '_hA'), 'hA', range(6))

    # for n, img_list in zip(name_list, image_list):
        # image_utils.save_gif(img_list, osp.join(save_dir, name + '_%s' % n))

    file_list = [osp.join(save_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list

def run_gt(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6):

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

@torch.no_grad()
def run_method_fig(dataloader, trainer: VolSDFHoi, save_dir, name, H, W, render_kwargs_test={}):
    device = trainer.device
    if isinstance(trainer, DistributedDataParallel):
        trainer = trainer.module
    os.makedirs(save_dir, exist_ok=True)

    
    # reconstruct  hand and render
    for i, (indices, model_input, ground_truth) in enumerate(tqdm(dataloader)):
        model_input = model_utils.to_cuda(model_input)
        ground_truth = model_utils.to_cuda(ground_truth)
        # hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        # gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        # novel_img = trainer.render_novel_view(val_ind, val_in, val_gt, render_kwargs_test)
        # loss_extras = trainer(trainer.args, indices, model_input, ground_truth, render_kwargs_test, 0)
        # ret = loss_extras['extras']
        
        H = 512
        gt = torch.cat([
            model_input['hand_mask'].reshape(1, 1, H, H), 
            model_input['obj_mask'].reshape(1, 1, H, H), 
            torch.zeros_like(model_input['obj_mask']).reshape(1, 1, H, H)], 1)
        print(gt.shape, ground_truth['rgb'].shape)
        print(osp.join(save_dir,  f'methodfig/{i}_gt.png'))
        image_utils.save_images(gt, osp.join(save_dir,  f'methodfig/{i}_gt.png'))
        image_utils.save_images(ground_truth['rgb'].reshape(1, H, H, 3).permute(0, 3, 1, 2), osp.join(save_dir,  f'methodfig/{i}_inp.png'))

@torch.no_grad()
def run_fig(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6, max_t=1000, T_num=None):
    device = trainer.device
    model = trainer.model
    print(save_dir)
    renderer = partial(trainer.mesh_renderer, soft=False)

    mesh_file = osp.join(save_dir, name + '_obj.ply')
    if not osp.exists(mesh_file):
        # if True:
        try:
            mesh_util.extract_mesh(
                    model.implicit_surface, 
                    N=N,
                    filepath=mesh_file,
                    volume_size=volume_size,
                )
        except ValueError:
            jObj = plot_utils.create_coord('cpu', size=0.00001)
            mesh_utils.dump_meshes([mesh_file[:-4]], jObj)
            mesh_file = mesh_file.replace('.ply', '.obj')

    jObj = mesh_utils.load_mesh(mesh_file).cuda()
    # reconstruct HOI and render in origin, 45, 60, 90 degree
    degree_list = [0, 45, 60, 90, 180, 360-60, 360-90]
    name_list = ['gt', 'overlay_hoi', 'overlay_obj']
    for d in degree_list:
        name_list += ['%d_hoi' % d, '%d_obj' % d]  


    image_list = [[] for _ in name_list]
    T = len(dataloader)

    if T_num is None:
        T_list = [0, T//2, T-1]
    else:
        T_list = np.linspace(0, T-1, T_num).astype(np.int).tolist() 
    print('len', T, T_list)
    for t, (indices, model_input, ground_truth) in enumerate(dataloader):
        if t not in T_list:
            continue
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        gt = F.adaptive_avg_pool2d(gt, (H, W))

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        K_ndc = mesh_utils.intr_from_screen_to_ndc(intrinsics, H, W)
        hoi, obj = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTc, H=H, K_ndc=K_ndc)
        # image1, mask1 = render(renderer, jHand, jObj, jTc, intrinsics, H, W)
        image_list[0].append(gt)
        image_list[1].append(hoi)  # view 0
        image_list[2].append(obj)

        for i, az in enumerate(degree_list):
            img1, img2 = mesh_utils.render_hoi_obj(jHand, jObj, az, jTc=jTc, H=H, W=W)
            image_list[3 + 2*i].append(img1)  
            image_list[3 + 2*i+1].append(img2) 
        
        # save 
        for n, im_list in zip(name_list, image_list):
            im = im_list[-1]
            image_utils.save_images(im, osp.join(save_dir, f'{t:03d}_{name}_{n}'))

    

def run(dataloader, trainer, save_dir, name, H, W, offset=None, N=64, volume_size=6, max_t=1000):
    device = trainer.device
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 2]]).to(device), 
            )
    rot_y = geom_utils.axis_angle_t_to_matrix(
        np.pi / 2 * torch.FloatTensor([[1, 0, 0]]).to(device), 
    )
    if isinstance(trainer, DistributedDataParallel):
        trainer = trainer.module

    model = trainer.model
    renderer = partial(trainer.mesh_renderer, soft=False)

    os.makedirs(save_dir, exist_ok=True)
    # reconstruct object
    # if is_master():
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
    for i, (indices, model_input, ground_truth) in enumerate(dataloader):
        if i >= max_t:
            break
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        gt = F.adaptive_avg_pool2d(gt, (H, W))

        jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
            indices.to(device), model_input, ground_truth, H, W)
        mesh_utils.dump_meshes([osp.join(save_dir, name + '_jHand', '%d' % indices[0].item())], jHand)
        hTj = geom_utils.inverse_rt(mat=jTh, return_mat=True)
        image1, mask1 = render(renderer, jHand, jObj, jTc, intrinsics, H, W)
        
        cam_norm = mesh_utils.get_camera_dist(wTc=jTc)
        xyz = torch.cat([torch.zeros_like(cam_norm), torch.zeros_like(cam_norm), -cam_norm * 0.1])
        z_back = geom_utils.axis_angle_t_to_matrix(t=xyz)
        print(jHand.device, jObj.device, jTc.device, H, W)
        image2, _ = render(renderer, jHand, jObj, jTc@z_back@offset, intrinsics, H, W)
        image3, _ = render(renderer, jHand, jObj, None, None, H, W)
        image4, _ = render(renderer, 
            mesh_utils.apply_transform(jHand, hTj), 
            mesh_utils.apply_transform(jObj, hTj), None, None, H, W)
        image5, _ = render(renderer, 
            mesh_utils.apply_transform(jHand,  rot_y@hTj), 
            mesh_utils.apply_transform(jObj, rot_y@hTj), None, None, H, W)

        image_list[0].append(gt)
        # print(gt.shape, image1.shape, mask1.shape)
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

def load_model_data(load_pt, H=224, W=224):
    device = 'cuda:0'
    # load config
    config = io_util.load_yaml(osp.join(load_pt.split('/ckpts')[0], 'config.yaml'))
    # print('###### Change it back! latter',  )

    # load data    
    dataset, _ = get_data(config, return_val=True, val_downscale=1)
    dataloader = DataLoader(dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mesh_utils.collate_meshes)
    # build and load model 
    posenet, focal_net = get_camera(config, datasize=len(dataset)+1, H=dataset.H, W=dataset.W)
    model, trainer, render_kwargs_train, render_kwargs_test, _, _ = get_model(config, data_size=len(dataset)+1, cam_norm=dataset.max_cam_norm, device=[0])
    render_kwargs_test['H'] = H
    render_kwargs_test['W'] = W

    state_dict = torch.load(load_pt)
    
    model_utils.load_my_state_dict(model, state_dict['model'])
    model_utils.load_my_state_dict(posenet, state_dict['posenet'])
    model_utils.load_my_state_dict(focal_net, state_dict['focalnet'])
    # model.load_state_dict(state_dict['model'])
    # posenet.load_state_dict(state_dict['posenet'])
    # focal_net.load_state_dict(state_dict['focalnet'])
    
    trainer.train_dataloader = dataloader
    trainer.val_dataloader = dataloader

    trainer.init_camera(posenet, focal_net)
    trainer.to(device)
    trainer.eval()

    return trainer, dataloader


@hydra.main(config_path='../configs', config_name='eval')
def main_batch(args):
    slurm_utils.update_pythonpath_relative_hydra()
    print(osp.join(args.load_dir + '*', 'ckpts/latest.pt'))
    model_list = glob(osp.join(args.load_dir + '*', 'ckpts/latest.pt'))
    for e in model_list:
        args.load_pt = e
        print(args.load_pt)
        try:
            main_function(args)
        except ValueError as er:
            print(e, er)
            continue

def main_function(args):
    # slurm_utils.update_pythonpath_relative_hydra()

    H = W = args.reso
    device = 'cuda:0'
    # args.load_pt = 
    # load config
    config = io_util.load_yaml(osp.join(args.load_pt.split('/ckpts')[0], 'config.yaml'))
    # print('###### Change it back! latter',  )
    # config.novel_view.mode = 'geom'
    # config.novel_view.amodal = 'amodal'
    # config.novel_view.diffuse_ckpt = '/home/yufeiy2/scratch/result/vhoi/geom/ho3d_cam_train_seg/checkpoints/last.ckpt'

    # load data    
    dataset, _ = get_data(config, return_val=True, val_downscale=1)
    dataloader = DataLoader(dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mesh_utils.collate_meshes)
    print(len(dataset), len(dataloader))

    # build and load model 
    posenet, focal_net = get_camera(config, datasize=len(dataset)+1, H=dataset.H, W=dataset.W)
    model, trainer, render_kwargs_train, render_kwargs_test, _, _ = get_model(config, data_size=len(dataset)+1, cam_norm=dataset.max_cam_norm, device=[0])
    trainer.train_dataloader = dataloader
    render_kwargs_test['H'] = H
    render_kwargs_test['W'] = W

    assert args.out is not None or args.load_pt is not None, 'Need to specify one of out / load_pt'
    
    if args.load_pt is not None:
        state_dict = torch.load(args.load_pt)
        
        model_utils.load_my_state_dict(model, state_dict['model'])
        model_utils.load_my_state_dict(posenet, state_dict['posenet'])
        model_utils.load_my_state_dict(focal_net, state_dict['focalnet'])
        # model.load_state_dict(state_dict['model'])
        # posenet.load_state_dict(state_dict['posenet'])
        # focal_net.load_state_dict(state_dict['focalnet'])
        
        trainer.init_camera(posenet, focal_net)
        trainer.to(device)
        trainer.eval()
        # trainer.train()
        
        it = state_dict['global_step']
        name = 'it%08d' % it
        save_dir = args.load_pt.split('/ckpts')[0] + '/vis_clip/'
    
    if args.out is not None:
        save_dir = args.out

    os.makedirs(save_dir, exist_ok=True)

    if args.gt:
        with torch.no_grad():
            run_gt(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N)
    if args.surface:
        with torch.no_grad():
            run(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N)
    if args.method_fig:
        render_kwargs_test['rayschunk'] = args.chunk
        render_kwargs_test['H'] = H #  * 2 // 3
        render_kwargs_test['W'] = W #  * 2 // 3

        with torch.no_grad():
            run_method_fig(dataloader, trainer, save_dir, name, H=H, W=W, render_kwargs_test=render_kwargs_test)
    if args.fig:
        with torch.no_grad():
            run_fig(dataloader, trainer, save_dir, name, H=H, W=W, volume_size=args.volume_size, N=args.N, T_num=args.T_num)
    if args.video:
        with torch.no_grad():
            save_dir = args.load_pt.split('/ckpts')[0]
            run_video(dataloader, trainer, save_dir, 'vis_video', H=H, W=W, volume_size=args.volume_size, N=args.N)
    if args.nvs:
        render_kwargs_test['rayschunk'] = args.chunk
        render_kwargs_test['H'] = H #  * 2 // 3
        render_kwargs_test['W'] = W #  * 2 // 3
        with torch.no_grad():
            run_render(dataloader, trainer, save_dir, name, render_kwargs_test,)
    if args.nvs_train:
        render_kwargs_train['rayschunk'] = args.chunk
        render_kwargs_train['H'] = H #  * 2 // 3
        render_kwargs_train['W'] = W #  * 2 // 3
        render_kwargs_train['N_samples'] = args.D
        render_kwargs_train['max_upsample_steps'] = 1
        with torch.no_grad():
            run_train_render(dataloader, trainer, save_dir, name, render_kwargs_train,)
    if args.hand:
        with torch.no_grad():
            save_dir = args.load_pt.split('/ckpts')[0] + '/hA/'
            run_hA(dataloader, trainer, save_dir, name, )
    
    if args.cam:
        with torch.no_grad():
            render_kwargs_test['H'] = H
            render_kwargs_test['W'] = W
            run_vis_cam(dataloader, trainer, save_dir, name, render_kwargs_test)

    if args.diff:
        render_kwargs_train['rayschunk'] = args.chunk
        render_kwargs_train['H'] = H #  * 2 // 3
        render_kwargs_train['W'] = W #  * 2 // 3
        render_kwargs_train['N_samples'] = args.D
        render_kwargs_train['max_upsample_steps'] = 1
        with torch.no_grad():
            run_vis_diffusion(dataloader, trainer, save_dir, name, render_kwargs_train)


def render_az(jHand, jObj, jTc, az, H, W, f=2):
    jObj.textures = mesh_utils.pad_texture(jObj, 'white')
    if jHand is not None:
        jHand.textures = mesh_utils.pad_texture(jHand, 'blue')
        jMeshes = mesh_utils.join_scene([jHand, jObj])
    else:
        jMeshes = jObj
    cMeshes = mesh_utils.apply_transform(
        jMeshes, geom_utils.inverse_rt(mat=jTc, return_mat=True))

    image = mesh_utils.render_geom_azel(cMeshes, [az,0], out_size=H, f=f)
    return image['image'], image['mask']
    

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
            uv_mode=False,
            )
        image = iMesh['image']
        mask = iMesh['mask']
    return image.cpu(), mask.cpu()


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--out", type=str, default=None, help='output ply file name')
    # parser.add_argument('--reso', type=int, default=224, help='resolution of images')
    # parser.add_argument('--N', type=int, default=64, help='resolution of the marching cube algo')
    # parser.add_argument('--D', type=int, default=128, help='resolution of the marching cube algo')
    # parser.add_argument('--T', type=int, default=100, help='resolution of the marching cube algo')
    # parser.add_argument('--volume_size', type=float, default=6., help='voxel size to run marching cube')

    # parser.add_argument("--diff_ckpt", type=str, default='', help='the trained model checkpoint .pt file')
    # parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    # parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    # parser.add_argument("--chunk", type=int, default=256, help='net chunk when querying the network. change for smaller GPU memory.')

    # parser.add_argument("--hand", action='store_true')
    # parser.add_argument("--nvs", action='store_true')
    # parser.add_argument("--nvs_train", action='store_true')
    # parser.add_argument("--gt", action='store_true')
    # parser.add_argument("--surface", action='store_true')
    # parser.add_argument("--cam", action='store_true')
    # parser.add_argument("--diff", action='store_true')
    # args = parser.parse_args()
    
    main_batch()