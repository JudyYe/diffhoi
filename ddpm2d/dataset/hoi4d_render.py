import os.path as osp
from glob import glob
from .ho3d_render import get_hand_image, get_obj_image, get_image, sem_mask, read_depth, surface_normal

def parse_data(data_dir, split, data_cfg, args):
    """_summary_

    :param data_dir: _description_
    :param split: _description_
    :param data_cfg: _description_
    :param args: _description_
    :return: format according to base_data.py
        'image': list of image files
        'text': list of str
        'img_func': 
        'meta': {}
    """
    img_list = glob(f'{data_dir}/amodal_mask/*.png')
    index_list = [osp.basename(e)[:-4] for e in img_list]

    image_list = []
    meta = {
        'hand_depth_list': [],
        'obj_depth_list': [],
        'hand_normal_list': [],
        'obj_normal_list': [],
    }
    for index in index_list:
        img_file = f'{data_dir}/{{}}/{index}.png'
        image_list.append(img_file.format('amodal_mask'))
        meta['hand_depth_list'].append(img_file.format('hand_depth'))
        meta['obj_depth_list'].append(img_file.format('obj_depth'))
        meta['hand_normal_list'].append(img_file.format('hand_normal')[:-3] + 'npy')
        meta['obj_normal_list'].append(img_file.format('obj_normal')[:-3] + 'npy')
        
    text_list = ['a semantic segmentation of a hand grasping an object'] * len(image_list)
    print(args.mode.cond)
    # print(args.mode.out)
    if not args.mode.cond:
        img_func = get_image
        cond_func = None
    else:
        img_func = get_obj_image
        cond_func = get_hand_image

    if 
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
    import torch
    out = parse_data('/home/yufeiy2/scratch//data/HOI4D/amodal/', 
        '', {}, AttrDict({'zfar': 1, 'mode': {'cond': False}}))
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
