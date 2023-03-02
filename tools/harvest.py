from tqdm import tqdm
import numpy as np
from glob import glob
import os
import os.path as osp
from jutils import web_utils, mesh_utils
import tools.icp_recon as icp_tool

device = 'cuda:0'
cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
ind_list = [1,2]
index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]
data_dir = '/home/yufeiy2/scratch/result/HOI4D'


def get_iter(exp_dir, index):
    if args.method == 'vhoi':
        mesh_list = sorted(glob(osp.join(exp_dir, 'meshes', '*_obj.ply')))
        it_list = [osp.basename(m).split('.')[0].split('_')[0] for m in mesh_list]
        th = args.th
        # it_list = [it for it in it_list if int(it) < th]
        mesh_list= [m for m, it in zip(mesh_list, it_list) if int(it) < th]
        it_list = [it for it in it_list if int(it) < th]

        # filter out when it_list > th
        it = osp.basename(mesh_list[-1]).split('.')[0].split('_')[0]
    elif args.method == 'hhor':
        mesh_list = sorted(glob(osp.join(exp_dir, 'handscanner/checkpoints', '*.pth')))
        if len(mesh_list) > 0:
            it = osp.basename(mesh_list[-1]).split('.')[0].split('_')[1]
        else:
            it = -1

    return f'iter: {it}'

def get_gt_mesh(index):
    if args.data == 'hoi4d':
        mesh_file = osp.join(data_dir, index, 'oObj.obj')
        meshes = mesh_utils.load_mesh(mesh_file, device=device)
    return meshes

def eval_mesh(exp_dir, index):
    if args.method == 'vhoi':
        mesh_list = sorted(glob(osp.join(exp_dir, 'meshes', '*_obj.ply')))
        it_list = [osp.basename(m).split('.')[0].split('_')[0] for m in mesh_list]
        th = args.th
        mesh_list= [m for m, it in zip(mesh_list, it_list) if int(it) < th]
        it_list = [it for it in it_list if int(it) < th]
        if len(mesh_list) == 0:
            return 'iter: -1'
        it = osp.basename(mesh_list[-1]).split('.')[0].split('_')[0]
        mesh_file = mesh_list[-1]
    sources = mesh_utils.load_mesh(mesh_file, device=device,scale_verts=0.1)
    targets = get_gt_mesh(index)

    new_s, _ = icp_tool.register_meshes(sources, targets, scale=args.scale, seed=123)
    th_list = np.array([5, 10, 20]) * 1e-3
    f_list = mesh_utils.fscore(new_s, targets, th=th_list)
    f_list = np.array(f_list).reshape(-1)
    # cd is sum of squared chamfer distance
    f_list[-1] *= 1e4

    # convert list of list to str for pretty print in one line 
    f_str = ' '.join([f'{f:.3f}' for f in f_list])
    rtn = f'iter: {it} ' + f_str
    return rtn

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
    
    name = f'{args.type}'
    if args.type == 'progress':
        func = get_iter
    elif args.type == 'eval':
        func = eval_mesh
        name += f'_scale{args.scale}'
    for i, index in enumerate(tqdm(index_list, desc='instances')):
        # filter exp_list that contains index
        exp_list_i = [e for e in exp_list if index in e]
        if len(exp_list_i) > N:
            print(exp_list_i)

        for m, exp in enumerate(exp_list_i):
            rtn = func(exp, index)
            cell_list[i+1][m+1] = f'{osp.basename(exp):20}: ' + rtn

    # pretty print cell_list
    for row in cell_list:
        print(' '.join([f'{c:10}' for c in row]))

    web_utils.run(osp.join(args.dir, 'vis', f'{name}.html'), cell_list)
    return

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/yufeiy2/scratch/result/vhoi/pred_no_prior')
    parser.add_argument('--method', type=str, default='vhoi')
    parser.add_argument('--th', type=int, default=10000000000)
    parser.add_argument('--type', type=str, default='progress')
    parser.add_argument('--data', type=str, default='hoi4d')
    parser.add_argument('--hoi4d_tab', action='store_true')
    parser.add_argument('--scale', action='store_true')
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    
    if args.hoi4d_tab:
        main_tab()
