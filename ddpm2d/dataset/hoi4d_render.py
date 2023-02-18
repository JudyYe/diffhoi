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
    if args.hoi4d.split is None:
        img_list = glob(f'{data_dir}/amodal_mask/*.png')
    else:
        img_list = glob(f'{data_dir}/amodal_mask/*{args.hoi4d.split}*.png')
    index_list = [osp.basename(e)[:-4] for e in img_list]

    image_list = []
    meta = {
        'hand_depth_list': [],
        'obj_depth_list': [],
        'hand_normal_list': [],
        'obj_normal_list': [],
        'hand_uv_list': [],

    }
    for index in index_list:
        img_file = f'{data_dir}/{{}}/{index}.png'
        image_list.append(img_file.format('amodal_mask'))
        meta['hand_depth_list'].append(img_file.format('hand_depth'))
        meta['obj_depth_list'].append(img_file.format('obj_depth'))
        meta['hand_normal_list'].append(img_file.format('hand_normal')[:-3] + 'npz')
        meta['obj_normal_list'].append(img_file.format('obj_normal')[:-3] + 'npz')
        meta['hand_uv_list'].append(img_file.format('hand_uv')[:-3] + 'npz')
        
    if args.cat_level:
        mapping = [
            '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
            'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
            'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
        ]
        text_list = []
        for i in range(len(image_list)):
            index = osp.basename(image_list[i])
            c = mapping[int(index.split('_')[2][1:])].lower()
            text_list.append(f'an image of a hand grasping a {c}')
    else:
        text_list = ['a semantic segmentation of a hand grasping an object'] * len(image_list)
    print(args.mode.cond)
    # print(args.mode.out)
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
    import pickle
    import torch
    out = parse_data('/home/yufeiy2/scratch/result/HOI4D/vis/', '', 
        {}, AttrDict({'zfar': 1, 'mode': {'cond': True, 'uv': True}, 'hoi4d': {'split': 'ZY20210800001_H1_C2_N31_S92_s'}}))
    # out = parse_data('/home/yufeiy2/scratch//data/HOI4D/amodal/', 
    #     '', {}, AttrDict({'zfar': 1, 'mode': {'cond': True, 'uv': True}}))
    save_dir = '/home/yufeiy2/scratch/result/vis'
    from ddpm2d.models.glide import GeomGlide, CondGeomGlide
    from jutils import image_utils
    # geom_glide = GeomGlide(AttrDict({'exp_dir': save_dir, 'ndim': 11, 'side_x': 56, 'side_y': 56, 
    #                                  'mode': {'cond': False, 'mask': True, 'normal': True, 'depth': True}}))

    geom_glide = CondGeomGlide(AttrDict({'exp_dir': save_dir, 'ndim': 11, 'side_x': 56, 'side_y': 56, 
                                     'mode': {'cond': False, 'mask': True, 'normal': True, 'depth': True}}))


    mean_list, var_list = [], []
    for i in range(10):
        img = get_obj_image(out['image'][i], i, out['meta'])
        cond = get_hand_image(out['image'][i], i, out['meta'])
        print(img.shape, cond.shape)
        with open('/home/yufeiy2/scratch/result/vis_ddpm/input/handuv.pkl', 'wb') as fp:
            pickle.dump({'image': img[None], 'cond_image': cond[None]}, fp)

    #     mean_list.append(img.reshape(11, -1).mean(-1),)
    #     var_list.append( img.reshape(11, -1).std(-1))
    #     img = img[None]

    #     rtn = geom_glide.decode_samples(img)

    #     for k, v in rtn.items():
    #         if 'normal' in k:
    #             v = v /2 + 0.5
    #         if 'depth' in k:
    #             print(k, v.max(), v.min(), )
    #             image_utils.save_depth(v.clip(-1, 10), osp.join(save_dir, f'{i}_{k}'), znear=-1, zfar=1)
    #             image_utils.save_images(v.clip(-1, 10), osp.join(save_dir, f'{i}_{k}2'), )
    #         else:
    #             image_utils.save_images(v.clip(-1, 1), osp.join(save_dir, f'{i}_{k}'))

    # print('mean', torch.stack(mean_list).mean(0))
    # print('std' , torch.stack(var_list).mean(0))
