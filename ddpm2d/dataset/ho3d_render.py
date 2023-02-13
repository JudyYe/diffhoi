import numpy as np
from PIL import Image
from glob import glob
import os.path as osp
from ..utils.train_util import pil_image_to_norm_tensor

import torch

def get_hand_image(image_file, ind, meta):
    mask = sem_mask(image_file, ind, meta)
    normal = surface_normal(image_file, ind, meta)
    depth = read_depth(image_file, ind, meta)

    image = torch.cat([mask[0:1], normal[0:3], depth[0:1]], 0)
    if meta['cfg'].mode.uv:
        uv = read_uv(image_file, ind, meta)
        image = torch.cat([image, uv], 0)
    return image


def get_obj_image(image_file, ind, meta):
    mask = sem_mask(image_file, ind, meta)
    normal = surface_normal(image_file, ind, meta)
    depth = read_depth(image_file, ind, meta)
    image = torch.cat([mask[1:2], normal[3:6], depth[1:2]], 0)
    return image


def get_image(image_file, ind, meta):
    mask = sem_mask(image_file, ind, meta)
    normal = surface_normal(image_file, ind, meta)
    depth = read_depth(image_file, ind, meta)
    image = torch.cat([mask, normal, depth], 0)
    return image


def sem_mask(image_file, ind, meta):
    """in HO3D rendereing, green object, red hand
    convert to -1 / 1
    """
    image = np.array(Image.open(image_file))
    out = np.zeros(image.shape)
    #R:
    out[..., 0] = (image[..., 0] > 50) * 255
    out[..., 1] = (image[..., 1] > 50) * 255
    
    out = out.astype(np.uint8)
    return pil_image_to_norm_tensor(Image.fromarray(out))


def read_depth(image_file, ind, meta):
    """
    input: R: hand, B: object
    get relative depth, in cm?, using hand as a scalar? 
    depth = 0: nan
    (obj - hand_center) / metric? 
    :param image_file: _description_
    :param ind: _description_
    :param meta: _description_
    :return: depth tensor in shape of [C=2, H, W], scale in dm
    """
    # saved depth in np.array is in mm. let us convert to dm/100mm
    cfg = meta['cfg']
    hand = np.array(Image.open(meta['hand_depth_list'][ind])).astype(np.float)
    hand_mask = hand > 0
    obj = np.array(Image.open(meta['obj_depth_list'][ind])).astype(np.float)
    obj_mask = obj > 0
    hand_mean = hand.sum() / hand_mask.sum()  # mean within mask
    z_far = cfg.zfar * 100  # 1m
    # TODO: only apply within mask
    obj = (obj - hand_mean) * obj_mask + z_far * (1 - obj_mask)
    hand = (hand - hand_mean) * hand_mask + z_far * (1 - hand_mask)

    # let us assume depth is in scale of 0, 200ish mm ? for now? 
    # the scene has a boundary so range is bounded, dont' need inverse depth trick  
    hoi_depth = np.stack([hand, obj], 0) # (C, H, W)
    hoi_depth = torch.FloatTensor(hoi_depth)
    # convert mm to d?m
    
    # 1 is 100mm
    hoi_depth /= 100
    return hoi_depth


def read_uv(image_file, ind, meta):
    hand_uv = np.load(meta['hand_uv_list'][ind])['array']
    hand_uv = torch.FloatTensor(hand_uv)
    return hand_uv


def surface_normal(image_file, ind, meta):
# https://github.com/autonomousvision/monosdf/blob/e6f923c4a9c319ca0c6b5c7fad7d0b1b32b7550f/code/datasets/scene_dataset.py
    """
    input: R: hand, B: object
    normal in camera coordinate for now. 
    # 0,0,0: nan
    :param image_file: _description_
    :param ind: _description_
    :param meta: _description_
    :return: normal tensor in shape of [C, H, W]
    """
    # legacy
    if osp.exists(meta['hand_normal_list'][ind]):
        hand_normal = np.load(meta['hand_normal_list'][ind])['array']
        obj_normal = np.load(meta['obj_normal_list'][ind])['array']
    else:
        hand_normal = np.load(meta['hand_normal_list'][ind][:-1] + 'y')
        obj_normal = np.load(meta['obj_normal_list'][ind][:-1] + 'y')

    # important as the output of omnidata is normalized
    hand_normal = hand_normal * 2. - 1.
    obj_normal = obj_normal * 2. - 1.

    normal = np.concatenate([hand_normal, obj_normal], axis=0)
    normal_tensor = torch.from_numpy(normal).float()
    return normal_tensor


def parse_data(data_dir, split, data_cfg, args):
    """let us try no crop first? 
    :param data_dir: _description_
    :param split: _description_
    :param args: _description_
    :return: format according to base_data.py
        'image': list of image files
        'text': list of str
        'img_func': 
        'meta': {}
    """
    # with open(osp.join(data_dir, 'Sets', split + '.txt')) as fp:
    with open(split) as fp:
        index_list = [line.strip() for line in fp]
    image_list = []
    meta = {
        'hand_depth_list': [],
        'obj_depth_list': [],
        'hand_normal_list': [],
        'obj_normal_list': [],
        'hand_uv_list': [],
    }
    for index in index_list:
        s, vid, f_index = index.split('/')
        # for suf in ['origin', 'novel']:
        for suf in ['novel']:
            img_file = f'{data_dir}/{{}}/{vid}_{f_index}_{suf}.png'
            image_list.append(img_file.format('amodal_mask'))
            meta['hand_depth_list'].append(img_file.format('hand_depth'))
            meta['obj_depth_list'].append(img_file.format('obj_depth'))
            meta['hand_normal_list'].append(img_file.format('hand_normal')[:-3] + 'npz')
            meta['obj_normal_list'].append(img_file.format('obj_normal')[:-3] + 'npz')
            meta['hand_uv_list'].append(img_file.format('hand_uv')[:-3] + 'npz')
        
    text_list = ['a semantic segmentation of a hand grasping an object'] * len(image_list)
    print(args.mode.cond)
    if args.mode.cond == 0:
        img_func = get_image
        cond_func = None
    elif args.mode.cond == 1:
        img_func = get_obj_image
        cond_func = get_hand_image
    elif args.mode.cond == -1:
        img_func = get_obj_image
        cond_func = None

    meta['cfg'] = args
    return {
        'image': image_list,
        'text': text_list,
        'img_func': img_func, 
        'cond_func': cond_func,
        'meta': meta,
    }    


if __name__ == '__main__':
    from attrdict import AttrDict
    out = parse_data('/home/yufeiy2/scratch//data/HO3D/crop_render/', 
        '/home/yufeiy2/scratch//data/HO3D/Sets/SM2.txt', {}, AttrDict({'zfar': 1, 'mode': {'cond': False}}))
    save_dir = '/home/yufeiy2/scratch/result/vis'
    from ddpm2d.models.glide import GeomGlide
    from jutils import image_utils
    geom_glide = GeomGlide(AttrDict({'ndim': 11, 'side_x': 56, 'side_y': 56, 
                                     'mode': {'cond': False, 'mask': True, 'normal': True, 'depth': True}}))
    mean_list, var_list = [], []
    for i in range(10):
        img = get_image(out['image'][i], i, out['meta'])
        mean_list.append(img.reshape(11, -1).mean(-1),)
        var_list.append( img.reshape(11, -1).std(-1))
        img = img[None]

        rtn = geom_glide.decode_samples(img)

        for k, v in rtn.items():
            if 'normal' in k:
                v = v /2 + 0.5
            if 'depth' in k:
                print(k, v.max(), v.min(), )
                image_utils.save_depth(v.clip(-1, 10), osp.join(save_dir, f'{i}_{k}'), znear=-1, zfar=1)
                image_utils.save_images(v.clip(-1, 10), osp.join(save_dir, f'{i}_{k}2'), )
            else:
                image_utils.save_images(v.clip(-1, 1), osp.join(save_dir, f'{i}_{k}'))

    print('mean', torch.stack(mean_list).mean(0))
    print('std' , torch.stack(var_list).mean(0))
