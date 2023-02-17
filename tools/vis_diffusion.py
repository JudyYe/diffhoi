"""script to debug SDLoss"""
import pickle
import numpy as np
import os
import os.path as osp
from time import time
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from jutils import image_utils, model_utils
from models.sd import SDLoss

device = 'cuda:0'
def get_inp(image_file, cfg):
    if image_file.endswith('.pkl'):
        image = pickle.load(open(image_file, 'rb'))
        new_image = {}
        for k, v in image.items():
            image_list = []
            if cfg.mask:
                image_list.append(v[:, :1])
            if cfg.normal:
                image_list.append(v[:, 1:4])
            if cfg.depth:
                image_list.append(v[:, 4:5])
            if cfg.uv and v.shape[1] > 5:
                image_list.append(v[:, 5:7])
            new_image[k] = torch.cat(image_list, 1)

        image = model_utils.to_cuda(new_image)

    else:
        # TODO: emmm is GLIDe trained with scale [-1, 1] or [0, 1?]-->[-1, 1]
        image = Image.open(image_file)
        H = W = args.reso
        image = image.resize((H, W))
        image = ToTensor()(image)[:3]
        image = image * 2 - 1
        image = image[None]
        image = image.to(device)

    return image



def run_sample(sd: SDLoss, image_file):
    save_dir = args.out
    H = args.reso

    image = get_inp(image_file, sd.model.cfg.mode)

    name_list = ['noisy_img', 'multi_step', 'single_step', 'chain_single']
    image_list = [{} for _ in name_list]
    text_list = [[] for _ in name_list]

    pref = osp.basename(image_file)[:-4] + '_'

    ndim = sd.model.cfg.ndim
    noise = torch.randn([args.S, ndim, H, H], device=device)
    # each image is a num_sample x noise_level
    for noise_level in tqdm(args.noise.split(',')):
        noise_level = float(noise_level)
        if args.noise_is_step:
            noise_step = int(noise_level)
        else:
            noise_step = int(noise_level * sd.num_step)

        m_list, s_list, c_list, cd_list = [], [], [], []
        for s in tqdm(range(args.S)):
            # noisy = sd.get_noisy_image(image, noise_step-1, noise[s:s+1])
            noisy = sd.get_noisy_image(image['image'], noise_step-1, noise[s:s+1])
            single_out, extra = sd.vis_single_step(
                noisy, noise_step-1, noise=noise[s:s+1], cond_image=image.get('cond_image', None))
            print((extra['noise'] - noise[s:s+1]).norm())
            out = image['image']
            for t in list(range(noise_step))[::-1]:
                out = sd.get_noisy_image(out, t, noise[s:s+1])
                out, _ = sd.vis_single_step(
                    out, t, args.guidance, noise=noise[s:s+1], cond_image=image.get('cond_image', None))
            c_list.append(out)

            # multi_out = single_out
            multi_out = sd.vis_multi_step(
                noisy, noise_step, args.guidance,  loop=args.loop, cond_image=image.get('cond_image', None))
            # single_out = multi_out
            m_list.append(multi_out)
            s_list.append(single_out)
            if 'cond_image' in image:
                cd_list.append(image.get('cond_image'))
        multi_cond = torch.cat(cd_list, dim=-2) if len(cd_list) > 0 else None
        multi_out = torch.cat(m_list, dim=-2) # concat in height
        single_out = torch.cat(s_list, dim=-2)
        c_list = torch.cat(c_list, dim=-2)

        append(image_list[0], noisy, sd)
        append(image_list[1], multi_out, sd, multi_cond, )
        append(image_list[2], single_out, sd, multi_cond, )
        append(image_list[3], c_list, sd, multi_cond, )
        # image_list[0].append(noisy)
        # image_list[1].append(multi_out)
        # image_list[2].append(single_out)
        # image_list[3].append(c_list)

        text_list[1].append(f'm{noise_level:g}')
        text_list[2].append(f's{noise_level:g}')
        text_list[3].append(f'c{noise_level:g}')
    
    print(f'save image to {save_dir}') 
    for k, v in image.items():
        out = sd.model.decode_samples(v)
        for kk, vv in out.items():
            image_utils.save_images(vv, osp.join(save_dir, pref + 'inp'), scale=True)
    for name, images_dict, texts in zip(name_list, image_list, text_list):
        for k, images in images_dict.items():
            # [(1, C, H, W), ]
            if len(texts) == 0:
                texts.append(None)
            images = torch.cat(images)
            print(osp.join(save_dir, pref + name + f'_{k}'))
            image_utils.save_images(
                images,
                osp.join(save_dir, pref + name + f'_{k}'),
                texts, col=len(images), scale=True
            )


def append(image_list, image, sd: SDLoss, cond_image=None):
    out = sd.model.decode_samples(image)
    # out['obj_normal']
    # out['obj_depth']
    # out['semantics']
    if cond_image is not None:
        cond = sd.model.decode_samples(cond_image)
        if 'semantics' in out:
            out['semantics'] = torch.cat([out['semantics'], cond['semantics'], torch.zeros_like(out['semantics'])], 1)
    

    for k,v in out.items():
        if k not in image_list:
            image_list[k] = []
        
        if 'normal' in k:
            v = v / 2 + 0.5
        if 'depth' in k:
            color = image_utils.save_depth(v, None, znear=-1, zfar=1)
            color = TF.to_tensor(color)[None]
            if 'depth_color' not in image_list:
                image_list['depth_color'] = []
            image_list['depth_color'].append(color.cpu())
        print(k, v.shape)
        image_list[k].append(v.cpu())
    return image_list


def run_sgd(sd: SDLoss, image_file):
    save_dir = args.out
    H = args.reso
    pref = osp.basename(image_file)[:-4] + '_'

    image = get_inp(image_file)
    image = image.to(device)
    image = nn.Parameter(image)
    opt = optim.AdamW([image], lr=1e-2)

    print(f'save image to {save_dir}') 
    image_utils.save_images(image, osp.join(save_dir, pref + 'inp'), scale=True)

    name_list = ['sds', ]
    image_list = [[] for _ in name_list]
    text_list = [[] for _ in name_list]

    noise_level = float(args.noise.split(',')[0])
    if args.noise_is_step:
        noise_step = int(noise_level)
    else:
        noise_step = int(noise_level * sd.num_step)

    # for s in tqdm(range(args.S)):
    # line = []
    for t in list(range(noise_step))[::-1]:
        opt.zero_grad()
        t_tensor = torch.tensor([t], device=device)
        sd.apply_sd(image, 1, t=t_tensor, )
        print(image.grad.data.norm())
        opt.step()

        denoised = image.data.cpu().detach()
        image_list[0].append(denoised)
    # image_list[0].append(torch.cat(line, dim=-1))
    
    for name, images, texts in zip(name_list, image_list, text_list):
        # [(1, C, H, W), ]
        if len(texts) == 0:
            texts.append(None)
        images = torch.cat(images)
        print(images.shape, len(texts))
        image_utils.save_images(
            images,
            osp.join(save_dir, pref + name),
            texts, scale=True
        )
    

def main_function(args):
    torch.manual_seed(args.seed)
    save_dir = args.out
    os.makedirs(save_dir, exist_ok=True)
    
    sd = SDLoss(args.load_pt, prediction_respacing=args.total_step)
    sd.init_model(device)

    inp_image_list = sorted(glob(args.inp))

    for image_file in inp_image_list:

        if 'sample' in args.mode:
            run_sample(sd, image_file)
        if 'sgd' in args.mode:
            run_sgd(sd, image_file)
        if 'uncond' in args.mode:
            run_uncond_sample(sd)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="/home/yufeiy2/scratch/result/vis_ddpm/input/handuv.pkl", help='output ply file name')
    parser.add_argument("--out", type=str, default="/home/yufeiy2/scratch/result/vis_ddpm/", help='output ply file name')
    parser.add_argument("--mode", type=str, default="sample", help='output ply file name')
    parser.add_argument('--reso', type=int, default=64, help='resolution of images')
    parser.add_argument('--guidance', type=float, default=4, help='CFG')
    parser.add_argument('--noise', type=str, default='0.98,0.9,0.75,0.5,0.25,0.1', help='noise level')
    parser.add_argument('--total_step', type=int, default=100, help='total noise')
    parser.add_argument('--loop', type=str, default='ddim', help='sample loop')
    parser.add_argument('--S', type=int, default=8, help='number of samples')
    # parser.add_argument("--load_pt", type=str, default='/home/yufeiy2/scratch/result/vhoi/ddpm_novel_sunday/hoi4d_CondGeomGlide_1/checkpoints/last.ckpt', 
    #     help='the trained model checkpoint .pt file')
    parser.add_argument("--load_pt", type=str, default='ddpm_novel_sunday/hoi4d_CondGeomGlide_1', 
        help='the trained model checkpoint .pt file')
    parser.add_argument('--noise_is_step', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    args.load_pt = f'/home/yufeiy2/scratch/result/vhoi/{args.load_pt}/checkpoints/last.ckpt'
    
    main_function(args)