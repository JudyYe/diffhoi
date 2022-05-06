import torch
from torch.utils.data import DataLoader
from ddpm.data import SdfData
from engine import update_pythonpath_relative_hydra
from utils.hand_utils import ManopthWrapper, get_nTh
import os
import os.path as osp
from ddpm.main import load_diffusion_model
from jutils import mesh_utils, image_utils, geom_utils
from hydra import main
from utils import io_util


@main(config_path='./configs', config_name='vary_hand')
def test(args):

    device = 'cuda:0'
    args.ckpt = osp.join(args.common_dir, args.ckpt)
    diffusion = load_diffusion_model(args.ckpt).to(device)
    save_dir = args.ckpt.split('/ckpt/')[0] + '/eval_vary_hand/'
    os.makedirs(save_dir, exist_ok=True)

    valset = SdfData(args.split, args.data_dir)
    # same shape, diff hand
    loader = DataLoader(valset, args.bs)
    hand_wrapper = ManopthWrapper().to(device)

    idx = args.idx
    for d, data in enumerate(loader):
        if d == idx:
            break
    bs = args.bs
    origin_sdf = data['nSdf'][0:1].repeat(args.bs, 1, 1, 1, 1).to(device)
    hA = data['hA'].to(device)
    nTh = get_nTh(hA=hA, hand_wrapper=hand_wrapper)
    nHand, _ = hand_wrapper(nTh, hA)

    nObj = mesh_utils.batch_grid_to_meshes(origin_sdf.squeeze(1), bs)
    nHoi = mesh_utils.join_scene([nObj, nHand])
    image_list = mesh_utils.render_geom_rot(nHoi, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, '%02d_origin_%s' % (idx, args.split)))

    if args.T > 0:
        sdf_recon = diffusion.sample(args.bs, img=origin_sdf, hA=hA, t=args.T-1, q_sample=args.q_sample)

        nObj = mesh_utils.batch_grid_to_meshes(sdf_recon.squeeze(1), bs)
        nHoi = mesh_utils.join_scene([nObj, nHand])
        image_list = mesh_utils.render_geom_rot(nHoi, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%02d_hoi_T%d_%d_%s' % (idx, args.T, args.q_sample, args.split)))



if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--bs', type=int, default=8, help='resolution of the marching cube algo')
    # parser.add_argument('--T', type=int, default=100, help='resolution of the marching cube algo')
    # parser.add_argument("--ckpt", type=str, default=None, help='the trained model checkpoint .pt file')
    # parser.add_argument("--split", type=str, default='test_full', help='the trained model checkpoint .pt file')
    # parser.add_argument("--q_sample", action='store_true')
    # parser.add_argument("--data_dir", type=str, default='/glusterfs/yufeiy2/fair/data/obman/', help='the trained model checkpoint .pt file')
    # args = parser.parse_args()
    update_pythonpath_relative_hydra()    
    test()