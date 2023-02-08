from glob import glob
import os.path as osp
from jutils import hand_utils, image_utils, mesh_utils, geom_utils

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
    with open(osp.join(data_dir, 'Sets', split + '.txt')) as fp:
        index_list = [line.strip() for line in fp]
    image_list = []
    for index in index_list:
        s, vid, f_index = index.split('/')
        img_file = f'{data_dir}/{s}/{vid}/seg/{f_index}.jpg'
        image_list.append(img_file)
    text_list = ['a semantic segmentation of a hand grasping an object'] * len(image_list)
    fn = None
    return {
        'image': image_list,
        'text': text_list,
        'img_func': fn, 
        'meta': {},
    }    


def make_list():
    data_dir = '/home/yufeiy2/scratch/data/ObMan'
    
    # split_name = 'all_seg'
    # query = osp.join(data_dir, '*/*/seg/*.jpg')

    # split_name = 'train_seg'
    # query = osp.join(data_dir, 'train/*/seg/*.jpg')

    # split_name = 'eval_seg'
    # query = osp.join(data_dir, 'evaluation/*/seg/*.jpg')

    split_name = 'SM2'
    query = osp.join(data_dir, f'train/{split_name}/seg/*.jpg')

    image_list = sorted(glob(query))
    print(query, len(image_list))
    with open(osp.join(data_dir, 'Sets', split_name + '.txt'), 'w') as fp:
        for img_file in image_list:
            f = img_file.split('/')[-1][:-4]
            v = img_file.split('/')[-3]
            s = img_file.split('/')[-4]
            fp.write(f'{s}/{v}/{f}\n')


if __name__ == '__main__':
    make_list()