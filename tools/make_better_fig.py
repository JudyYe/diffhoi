from PIL import Image
import imageio
import argparse
import numpy as np
from glob import glob
import os
import os.path as osp
from jutils import web_utils


# degree_list = [0, 60, 90, 180, 270, 300]
data_dir = '/home/yufeiy2/scratch/result/org/'
# save_dir = '/home/yufeiy2/scratch/result/figs/'
fig_dir = '/home/yufeiy2/scratch/result/figs_row/'

def overlay(image, inp):
    alpha = image[:, :, 3:4] / 255.0
    image = image.astype(np.float32)
    inp = inp.astype(np.float32)
    image = image[..., 0:3] * alpha + inp * (1 - alpha)
    image = np.concatenate([image, np.ones_like(alpha) * 255], -1)
    return image.astype(np.uint8)

def get_fig_dir(method, index):
    # if method == 'hhor':
        # return osp.join(data_dir, method, index, 'handscanner/vis_clip')
    return osp.join(data_dir, method, index, 'vis_clip')
def merge_fig():
    method_list = args.method.split(',')
    # cp input to index_inp.png
    
    inp_dict = {}
    for index in index_list:
        method = method_list[0]
        fig_dir = get_fig_dir(method, index)
        img_list = sorted(glob(osp.join(fig_dir, f'*_gt.png')))
        t = min(args.t, len(img_list))
        if len(img_list) == 0:
            print('no image?', osp.join(fig_dir, f'*_gt.png'))
            continue
        row = imageio.imread(img_list[t])
        inp_dict[index] = row

        fname = osp.join(row_dir, f'{index}_input.png')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        imageio.imwrite(fname, row)

    for index in index_list:
        row_list = []
        for method in method_list:
            image_list = []
            fig_dir = get_fig_dir(method, index)
            for suf in args.suf.split(','):
                suf = '_' + suf
                for degree in args.degree.split(','):
                    img_list = sorted(glob(osp.join(fig_dir, f'*_{degree}{suf}.png')))
                    if len(img_list) == 0:
                        continue
                    t = min(args.t, len(img_list))
                    image = imageio.imread(img_list[t])
                    if 'overlay' in degree:
                        print('resize overlay')
                        image = resize_crop(image, 1)
                        # image = overlay(image, inp_dict[index])
                    image_list.append(image)
            if len(image_list) == 0:
                continue
            elif len(image_list) == 2:
                row = put_one_col(image_list)
            elif len(image_list) == 4:
                row = put_to_2x2(image_list)
            else:
                row = put_one_row(image_list)
            row_list.append(row)
        if len(row_list) == 0:
            continue
        row = put_one_row(row_list)
        name = ','.join(method_list)
        fname = osp.join(row_dir, f'{index}_{name}.png')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        imageio.imwrite(fname, row)

    return

def resize_crop(image, perc=1.4):
    image = Image.fromarray(image)
    W, H = image.size
    new_W, new_H = int(W*perc), int(H*perc)
    m_W = (new_W - W) // 2
    m_H = (new_H - H) // 2
    image = image.resize((int(W*perc), int(H*perc)))
    center_box = [m_W, m_H, m_W+W, m_H+H]
    image = image.crop(center_box)
    return np.array(image)


def put_one_col(image_list):
    # 2 element, make 2x1 grid
    a = np.concatenate(image_list, axis=0)
    return a

def put_one_row(image_list):
    # 2 element, make 1x2 grid
    a = np.concatenate(image_list, axis=1)
    return a

def put_to_2x2(image_list):
    # 4 element,  make 2x2 grid
    N = len(image_list)
    a = np.concatenate(image_list[0:N//2], axis=0)
    b = np.concatenate(image_list[N//2:], axis=0)
    c = np.concatenate([a,b], axis=1)
    return c


def web_merge():
    web_dir = osp.join(row_dir, 'web')
    # empty_list of 100x10
    cell_list = [[[] for _ in range(10)] for _ in range(100)]
    cnt = 0
    for i, index in enumerate(index_list):
        line = []
        query = osp.join(row_dir, f'{index}_*.png')
        line = sorted(glob(query))
        if len(line) == 0:
            continue
        cnt += 1
        for j, fname in enumerate(line):
            cell_list[cnt][j] = fname
        # cell_list[i]
        # print(len(line), query)
        
        # cell_list.append(line)
        print(line)
    print(cell_list)
    web_utils.run(web_dir, cell_list, height=200)


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='hoi4d')
parser.add_argument('--degree', type=str, default='overlay,90')
parser.add_argument('--suf', type=str, default='hoi,obj')

parser.add_argument('--t', type=int, default=1)
parser.add_argument('--method', type=str, default=None)
parser.add_argument('--fig', type=str, default='default')
args = parser.parse_args()
row_dir = osp.join(fig_dir, args.fig)


def get_data_list(data):
    if data == 'hoi4d':
        data = 'hoi4d'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = [1,2]
        index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]
    elif data == 'wild': 
        index_list = get_data_list('3rd') + get_data_list('1st') + get_data_list('visor')
    elif data == '3rd':
        data = '3rd_nocrop'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = list(range(10))
        index_list = [f"{cat.lower()}{ind}" for ind in ind_list for cat in cat_list ]
    elif data == 'visor':
        data = 'VISOR'
        cat_list = "Kettle,Bowl,Knife,ToyCar".split(',')
        # ind_list = 
        index_list ='Kettle_101,Kettle_102,Bottle_102'.split(',')
    elif data == 'hhor':
        cat_list = 'AirPods,Doraemon'.split(',')
        ind_list = list(range(3))
        index_list = [f"{cat}_{ind}" for cat in cat_list for ind in ind_list  ]       

    elif data == '1st':
        data = '1st_nocrop'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = list(range(10))
        index_list = [f"{cat.lower()}_{ind}" for cat in cat_list for ind in ind_list  ]
    return index_list

index_list = get_data_list(args.data)

# cp_fig()

# cp_inp()
merge_fig()
web_merge()
# to_merge()
