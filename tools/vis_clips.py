import numpy as np
import os
import os.path as osp
from models.frameworks.volsdf_hoi import MeshRenderer
from utils import io_util, mesh_util
from utils.mesh_util import extract_mesh

from models.frameworks import get_model
from models.cameras import get_camera
from dataio import get_data

import torch
from torch.utils.data.dataloader import DataLoader
from jutils import image_utils, geom_utils, mesh_utils

H = W = 224
device = 'cuda:0'


def run(dataloader, trainer, save_dir, name, H, W, offset=None, ):
    if offset is None:
        offset = geom_utils.axis_angle_t_to_matrix(
            torch.FloatTensor([[0, 0, 1]]).to(device), 
            )
    model = trainer.model
    renderer = trainer.mesh_renderer

    os.makedirs(save_dir, exist_ok=True)
    # reconstruct object
    mesh_util.extract_mesh(
        model.implicit_surface, 
        N=args.N,
        filepath=osp.join(save_dir, name + '_obj.ply'),
        volume_size=args.volume_size,
    )
    jObj = mesh_utils.load_mesh(osp.join(save_dir, name + '_obj.ply')).to(device)

    # reconstruct  hand and render
    image1_list, image2_list, image3_list = [], [], []
    gt_list = []
    for (indices, model_input, ground_truth) in dataloader:
        hh = ww = int(np.sqrt(ground_truth['rgb'].size(1) ))
        gt = ground_truth['rgb'].reshape(1, hh, ww, 3).permute(0, 3, 1, 2)
        gt_list.append(gt)

        jHand, jTc, intrinsics = trainer.get_jHand_camera(indices, model_input, ground_truth, H, W)
        image1, mask1 = render(renderer, jHand, jObj, jTc, intrinsics, H, W)
        
        cam_norm = mesh_utils.get_camera_dist(wTc=jTc)
        xyz = torch.cat([torch.zeros_like(cam_norm), torch.zeros_like(cam_norm), -cam_norm * 0.1])
        z_back = geom_utils.axis_angle_t_to_matrix(t=xyz)

        image2, _ = render(renderer, jHand, jObj, jTc@z_back@offset, intrinsics, H, W)
        image3, _ = render(renderer, jHand, jObj, None, None, H, W)

        image1_list.append(image_utils.blend_images(image1, gt, mask1))
        image2_list.append(image2)
        image3_list.append(image3)

        # toto: render novel view!

    image_utils.save_gif(image1_list, osp.join(save_dir, name + '_view_0'))
    image_utils.save_gif(image2_list, osp.join(save_dir, name + '_view_1'))
    image_utils.save_gif(image3_list, osp.join(save_dir, name + '_view_2'))

    image_utils.save_gif(gt_list, osp.join(save_dir, name + '_gt'))


def main_function(args):
    renderer = MeshRenderer().to(device)
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

    run(dataloader, trainer, save_dir, name, H=H, W=W)


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
    parser.add_argument('--volume_size', type=float, default=2., help='voxel size to run marching cube')
    parser.add_argument("--load_pt", type=str, default=None, help='the trained model checkpoint .pt file')
    parser.add_argument("--chunk", type=int, default=16*1024, help='net chunk when querying the network. change for smaller GPU memory.')
    parser.add_argument("--init_r", type=float, default=1.0, help='Optional. The init radius of the implicit surface.')
    args = parser.parse_args()
    
    main_function(args)