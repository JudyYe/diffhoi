import numpy as np
from glob import glob
import os
import os.path as osp
from jutils import web_utils


cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
ind_list = [1,2]
index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]



def get_iter(exp_dir):
    if args.method == 'vhoi':
        mesh_list = sorted(glob(osp.join(exp_dir, 'meshes', '*.ply')))
        it = osp.basename(mesh_list[-1]).split('.')[0].split('_')[0]
    elif args.method == 'hhor':
        mesh_list = sorted(glob(osp.join(exp_dir, 'handscanner/checkpoints', '*.pth')))
        if len(mesh_list) > 0:
            it = osp.basename(mesh_list[-1]).split('.')[0].split('_')[1]
        else:
            it = -1

    return f'iter: {it}'


def get_exp_list():
    if args.method == 'vhoi':
        exp_list = sorted(glob(osp.join(args.dir, '*', 'ckpts', )))
        exp_list = [osp.dirname(e) for e in exp_list]
    elif args.method == 'hhor':
        exp_list = sorted(glob(osp.join(args.dir, '*', 'handscanner')))
        exp_list = [osp.dirname(e) for e in exp_list]
    return exp_list

def get_index_list(exp_list):
    if args.data == 'hoi4d':
        rtn_list = index_list
    elif args.data == 'hhor':
        rtn_list = [osp.basename(e) for e in exp_list]
    return rtn_list

def main_tab():
    exp_list = get_exp_list()
    N = 10 # int(np.ceil(len(exp_list) / len(index_list)))
    index_list = get_index_list(exp_list)
    # create 2D list cell_list in shape of (len(index_list), N)
    cell_list = [['' for _ in range(N)] for _ in range(len(index_list)+1)]
    
    # add index_list to the first col
    for i, index in enumerate(index_list):
        cell_list[i+1][0] = index
    if args.type == 'progress':
        func = get_iter
    for i, index in enumerate(index_list):
        # filter exp_list that contains index
        exp_list_i = [e for e in exp_list if index in e]
        if len(exp_list_i) > N:
            print(exp_list_i)

        for m, exp in enumerate(exp_list_i):
            rtn = func(exp)
            cell_list[i+1][m+1] = f'{osp.basename(exp):20}: ' + rtn
    # pretty print cell_list
    for row in cell_list:
        print(' '.join([f'{c:10}' for c in row]))

    web_utils.run(osp.join(args.dir, 'vis', f'{args.type}.html'), cell_list)
    return

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/yufeiy2/scratch/result/vhoi/pred_no_prior')
    parser.add_argument('--method', type=str, default='vhoi')
    parser.add_argument('--type', type=str, default='progress')
    parser.add_argument('--data', type=str, default='hoi4d')
    parser.add_argument('--hoi4d_tab', action='store_true')
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    
    if args.hoi4d_tab:
        main_tab()
