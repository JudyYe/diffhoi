import imageio
import argparse
import numpy as np
from glob import glob
import os
import os.path as osp
from jutils import web_utils

exp2hoi4d_fig = {
    'ours': 'which_prior_w0.01_exp/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catTrue_cfgFalse',
    'ours_wild': 'wild/{}{}',
    'ihoi_wild': '../ihoi/light_mow/hoi4d/{}',

    'obj_prior': 'which_prior_w0.01_exp/{}_suf_smooth_100_ObjGeomGlide_cond_all_linear_catTrue_cfgFalse',
    'hand_prior': 'which_prior_w0.01_exp/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catFalse_cfgFalse',
    'no_prior': 'pred_no_prior/{}_suf_smooth_100',

    'w_normal': 'ablate_weight/{}_m1.0_n0_d1.0',
    'w_mask': 'ablate_weight/{}_m0_n1.0_d1.0',
    'w_depth': 'ablate_weight/{}_m1.0_n1.0_d0',
    'w_color': 'ablate_color/{}_rgb0',
    'anneal': 'which_prior_w0.01/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catTrue_cfgFalse',

    'ihoi': '../ihoi/light_mow/hoi4d/{}',
    'hhor': '../hhor/hoi4d_go/{}/handscanner',
    'gt': 'gt/{}',

}
# exp2hoi4d_fig.format(method)/vis_clip



degree_list = [0, 60, 90, 180, 270, 300]
data_dir = '/home/yufeiy2/scratch/result/vhoi'
save_dir = '/home/yufeiy2/scratch/result/figs/'
fig_dir = '/home/yufeiy2/scratch/result/figs_row/'

def cp_fig():
    # for key in ['anneal']:
    # for key in exp2hoi4d_fig:
    for key in args.method.split(','):
        for index in index_list:
            for degree in degree_list:
                for suf in ['_hoi', '_obj',]:                        
                    if 'wild' in key:
                        query = osp.join(data_dir, exp2hoi4d_fig[key].format(data,index), 'vis_clip', f'*_{degree}{suf}.png')
                    else:
                        query = osp.join(data_dir, exp2hoi4d_fig[key].format(index), 'vis_clip', f'*_{degree}{suf}.png')
                    src_list = glob(query)
                    if len(src_list) == 0:
                        print(query)
                    for src_file in src_list:
                        t = osp.basename(src_file).split('_')[0]
                        if key == 'hhor':
                            t = osp.basename(src_file).split('_')[1]
                        dst = osp.join(save_dir, data, key, index, f'{t}_{degree}{suf}.png')
                        os.makedirs(osp.dirname(dst), exist_ok=True)
                        print(dst)
                        os.system(f'cp {src_file} {dst}')

def cp_inp():
    key = 'input'
    suf = '_gt'
    for index in index_list:
        # query = osp.join(data_dir, exp2hoi4d_fig['ours'].format(index), 'vis_clip', f'*{suf}.png')
        key = args.method.split(',')[0]
        if 'wild' in key:
            query = osp.join(data_dir, exp2hoi4d_fig[key].format(data,index), 'vis_clip', f'*{suf}.png')
        else:
            query = osp.join(data_dir, exp2hoi4d_fig[key].format(index), 'vis_clip', f'*{suf}.png')

        src_list = glob(query)
        if len(src_list) == 0:
            print(len(src_list), query)
        for src_file in src_list:
            t = osp.basename(src_file).split('_')[0]
            if key == 'hhor':
                t = osp.basename(src_file).split('_')[1]
            dst = osp.join(save_dir, data, 'input', index, f'{t}_inp.png')
            os.makedirs(osp.dirname(dst), exist_ok=True)
            os.system(f'cp {src_file} {dst}')
            print(dst)


def merge_fig():
    if args.method is None:
        method_list = exp2hoi4d_fig.keys()
    else:
        method_list = args.method.split(',')
    # cp input to index_inp.png
    for index in index_list:
        method = 'input'
        img_list = sorted(glob(osp.join(save_dir, args.data, method, index, f'*_inp.png')))
        t = min(args.t, len(img_list))
        if t == 0:
            print(osp.join(save_dir, args.data, method, index, f'*_inp.png'))
            continue
        row = imageio.imread(img_list[t])

        fname = osp.join(row_dir, f'{index}_{method}.png')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        print(fname)
        imageio.imwrite(fname, row)

    for index in index_list:
        row_list = []
        for method in method_list:
            image_list = []
            for suf in args.suf.split(','):
                suf = '_' + suf
                for degree in args.degree.split(','):
                    img_list = sorted(glob(osp.join(save_dir, args.data, method, index, f'*_{degree}{suf}.png')))
                    if len(img_list) == 0:
                        continue
                    t = min(args.t, len(img_list))
                    image_list.append(imageio.imread(img_list[t]))
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

def to_web():
    web_dir = osp.join(row_dir, 'web')
    method_list = ['input', 'gt', 'ours', 'ihoi', 'hhor', 'no_prior', 'obj_prior', 'hand_prior', 'w_mask', 'w_normal', 'w_depth', 'anneal']
    cell_list = []
    for index in index_list:
        line = []
        for method in method_list:
            fname = osp.join(row_dir, f'{index}_{method}.png')
            line.append(fname)
        cell_list.append(line)
    web_utils.run(web_dir, cell_list, height=200)


def web_merge():
    web_dir = osp.join(row_dir, 'web')
    cell_list = []
    for index in index_list:
        line = []
        query = osp.join(row_dir, f'{index}_*.png')
        a_list = sorted(glob(query))
        for fname in a_list:
            line.append(fname)
        cell_list.append(line)
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

if args.data == 'hoi4d':
    data = 'hoi4d'
    cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
    ind_list = [1,2]
    index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]
elif args.data == '3rd':
    data = '3rd_nocrop'
    cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
    ind_list = list(range(10))
    index_list = [f"{cat.lower()}{ind}" for ind in ind_list for cat in cat_list ]
elif args.data == 'visor':
    data = 'VISOR'
    cat_list = "Kettle,Bowl,Knife,ToyCar".split(',')
    # ind_list = 
    index_list ='Kettle_101,Kettle_102,Bottle_102'.split(',')
elif args.data == '1st':
    data = '1st_nocrop'
    cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
    ind_list = list(range(10))
    index_list = [f"{cat.lower()}_{ind}" for ind in ind_list for cat in cat_list ]

# cp_fig()

cp_inp()
merge_fig()
# web_merge()
# to_merge()
