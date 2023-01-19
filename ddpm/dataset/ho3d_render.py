import numpy as np
from PIL import Image
from glob import glob
import os.path as osp
from ..utils.train_util import pil_image_to_norm_tensor

import torch


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
    :return: depth tensor in shape of [C=2, H, W], scale in cm
    """
    hand = np.array(Image.open(meta['hand_depth_list'][ind])).astype(np.float)
    obj = np.array(Image.open(meta['obj_depth_list'][ind])).astype(np.float)
    hand_mean = hand.sum() / (hand > 0).sum()  # mean within mask
    obj = obj - hand_mean
    hand = hand - hand_mean
    # let us assume depth is in scale of 0, 255? for now? 
    # the scene has a boundary so range is bounded, dont' need inverse depth trick  

    hoi_depth = np.stack([hand, obj], 0) # (C, H, W)
    hoi_depth = torch.FloatTensor(hoi_depth)
    # convert mm to cm
    
    # 1 is 100mm
    hoi_depth /= 100
    return hoi_depth


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
    hand_normal = np.load(meta['hand_normal_list'][ind])
    obj_normal = np.load(meta['obj_normal_list'][ind])

    # important as the output of omnidata is normalized
    hand_normal = hand_normal * 2. - 1.
    obj_normal = obj_normal * 2. - 1.

    normal = np.concatenate([hand_normal, obj_normal], axis=0)
    normal_tensor = torch.from_numpy(normal).float()
    return normal_tensor


def parse_data(data_dir, split, args):
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
    }
    for index in index_list:
        s, vid, f_index = index.split('/')
        for suf in ['origin', 'novel']:
            img_file = f'{data_dir}/{{}}/{vid}_{f_index}_{suf}.png'
            image_list.append(img_file.format('amodal_mask'))
            meta['hand_depth_list'].append(img_file.format('hand_depth'))
            meta['obj_depth_list'].append(img_file.format('obj_depth'))
            meta['hand_normal_list'].append(img_file.format('hand_normal')[:-3] + 'npy')
            meta['obj_normal_list'].append(img_file.format('obj_normal')[:-3] + 'npy')
        
    text_list = ['a semantic segmentation of a hand grasping an object'] * len(image_list)
    img_func = get_image

    return {
        'image': image_list,
        'text': text_list,
        'img_func': img_func, 
        'meta': meta,
    }    


if __name__ == '__main__':
    out = parse_data('/home/yufeiy2/scratch//data/HO3D/crop_render/', 
        '/home/yufeiy2/scratch//data/HO3D/Sets/SM2.txt', {})

    mean_list, var_list = [], []
    for i in range(10):
        img = get_image(out['image'][i], i, out['meta'])
        mean_list.append(img.reshape(11, -1).mean(-1),)
        var_list.append( img.reshape(11, -1).std(-1))
        print(img.shape, )
    print('mean', torch.stack(mean_list).mean(0))
    print('std' , torch.stack(var_list).mean(0))

# mean tensor([-0.7618, -0.6442, -1.0000,  0.0081,  0.0032,  0.0794,  0.0076,  0.0096,
#          0.1470, -0.5141, -0.5287])
# std tensor([0.6338, 0.7510, 0.0000, 0.1502, 0.1650, 0.2270, 0.1617, 0.1292, 0.3159,
#         0.2056, 0.1374])