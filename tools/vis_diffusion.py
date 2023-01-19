"""script to debug SDLoss"""
import numpy as np
import os
import os.path as osp
from time import time
from tqdm import tqdm
from torchvision.transforms import ToTensor
from glob import glob
from tqdm import tqdm
import torch
from jutils import image_utils
from PIL import Image
from models.sd import SDLoss

device = 'cuda:0'
def get_inp(image_file):
    # TODO: emmm is GLIDe trained with scale [-1, 1] or [0, 1?]-->[-1, 1]
    image = Image.open(image_file)
    H = W = args.reso
    image = image.resize((H, W))
    image = ToTensor()(image)
    image = image * 2 - 1
    image = image[None]
    return image


def main_function(args):
    torch.manual_seed(args.seed)
    save_dir = args.out
    os.makedirs(save_dir, exist_ok=True)
    
    sd = SDLoss(args.load_pt, prediction_respacing=args.total_step)
    sd.init_model(device)

    H = args.reso
    inp_image_list = sorted(glob(args.inp))

    name_list = ['noisy_img', 'multi_step', 'single_step']
    for image_file in inp_image_list:
        image_list = [[] for _ in name_list]
        text_list = [[] for _ in name_list]

        pref = osp.basename(image_file)[:-4] + '_'
        image = get_inp(image_file)
        image = image.to(device)
        noise = torch.randn([args.S, 3, H, H], device=device)
        # each image is a num_sample x noise_level
        for noise_level in tqdm(args.noise.split(',')):
            noise_level = float(noise_level)
            if args.noise_is_step:
                noise_step = int(noise_level)
            else:
                noise_step = int(noise_level * sd.num_step)

            m_list, s_list = [], []
            for s in tqdm(range(args.S)):
                noisy = sd.get_noisy_image(image, noise_step-1, noise[s:s+1])
                single_out = sd.vis_single_step(noisy, noise_step)
                # multi_out = single_out
                multi_out = sd.vis_multi_step(noisy, noise_step, loop=args.loop)
                # single_out = multi_out
                m_list.append(multi_out)
                s_list.append(single_out)
            multi_out = torch.cat(m_list, dim=-2) # concat in height
            single_out = torch.cat(s_list, dim=-2)

            image_list[0].append(noisy)
            image_list[1].append(multi_out)
            image_list[2].append(single_out)

            text_list[1].append(f'm{noise_level:g}')
            text_list[2].append(f's{noise_level:g}')
        
        print(f'save image to {save_dir}') 
        image_utils.save_images(image, osp.join(save_dir, pref + 'inp'), scale=True)
        for name, images, texts in zip(name_list, image_list, text_list):
            # [(1, C, H, W), ]
            if len(texts) == 0:
                texts.append(None)
            images = torch.cat(images)
            print(images.shape, len(texts))
            image_utils.save_images(
                images,
                osp.join(save_dir, pref + name),
                texts, col=len(images), scale=True
            )


        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="/home/yufeiy2/scratch/result/vis_ddpm/input/*.png", help='output ply file name')
    parser.add_argument("--out", type=str, default="/home/yufeiy2/scratch/result/vis_ddpm/", help='output ply file name')
    parser.add_argument('--reso', type=int, default=64, help='resolution of images')
    parser.add_argument('--noise', type=str, default='0.9,0.75,0.5,0.25,0.1', help='noise level')
    parser.add_argument('--total_step', type=int, default=100, help='total noise')
    parser.add_argument('--loop', type=str, default='ddim', help='sample loop')
    parser.add_argument('--S', type=int, default=8, help='number of samples')
    parser.add_argument("--load_pt", type=str, default='/home/yufeiy2/scratch/result/vhoi/ddpm/glide_train_seg/checkpoints/last.ckpt', 
        help='the trained model checkpoint .pt file')
    parser.add_argument('--noise_is_step', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    main_function(args)