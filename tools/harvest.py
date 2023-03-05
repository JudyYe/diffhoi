import torch
from tqdm import tqdm
import numpy as np
from glob import glob
import os
import os.path as osp
from jutils import web_utils, mesh_utils, model_utils, geom_utils, image_utils, hand_utils
import tools.icp_recon as icp_tool
import tools.vis_clips as clip_tool
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


def get_pred_meshes(exp_dir):
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
    elif args.method == 'hhor':
        mesh_list = sorted(glob(osp.join(exp_dir, 'handscanner/meshes', '*_obj.ply')))
        it_list = [osp.basename(m).split('.')[0].split('_')[0] for m in mesh_list]
        th = args.th
        mesh_list= [m for m, it in zip(mesh_list, it_list) if int(it) < th]
        it_list = [it for it in it_list if int(it) < th]
        if len(mesh_list) == 0:
            return 'iter: -1'
        it = osp.basename(mesh_list[-1]).split('.')[0].split('_')[0]
        mesh_file = mesh_list[-1]
    elif args.method == 'ihoi':
        mesh_list = sorted(glob(osp.join(exp_dir, 'meshes', '*_hObj.obj')))
        mesh_file = mesh_list
        it = 10000
        
    return mesh_file, it
    

def get_predict_hTj(exp_dir,):
    """

    :param exp_dir: _description_
    :returns: List of hTj in shape of [(1, 4, 4), ]
    """
    trainer, valloader = clip_tool.load_model_data(osp.join(exp_dir, 'ckpts/latest.pt'))
    
    hTj_list, hA_list = [], []
    for val_ind, val_in, val_gt in valloader:
        val_in = model_utils.to_cuda(val_in, device)
        val_gt = model_utils.to_cuda(val_gt, device)
        val_ind = val_ind.to(device)
        jTc, jTc_n, jTh, jTh_n = trainer.get_jTc(val_ind, val_in, val_gt)
        hA = trainer.model.hA_net(val_ind, val_in, None)
        hTj_list.append(geom_utils.inverse_rt(mat=jTh, return_mat=True))
        hA_list.append(hA)
    return hTj_list, hA_list

def get_gt_hTo(index):
    if args.data == 'hoi4d':
        poses_dict = np.load(osp.join(data_dir, index, 'eval_poses.npz'))
        hTo_list = poses_dict['hTo']
        hA_gt = np.load(osp.join(data_dir, index, 'hands.npz'))['hA']
    hTo_list = [torch.FloatTensor(hTo).unsqueeze(0).to(device) for hTo in hTo_list]
    hA_list = [torch.FloatTensor(hA).unsqueeze(0).to(device) for hA in hA_gt]
    return hTo_list, hA_list

def get_pred_hSource(exp_dir):
    mesh_file, it = get_pred_meshes(exp_dir)
    hSource_list = []
    if args.method == 'vhoi':
        jSources = mesh_utils.load_mesh(mesh_file, device=device)
        hTj_list, hA_pred_list = get_predict_hTj(exp_dir)
        for hTj in hTj_list:
            hSource = mesh_utils.apply_transform(jSources, hTj)
            hSource = hSource
            hSource_list.append(hSource)
    elif args.method == 'hhor':
        mesh_list = sorted(glob(osp.join(exp_dir, 'handscanner/meshes', it, '*_obj_*.ply')))
        for mesh_file in mesh_list:
            hSource = mesh_utils.load_mesh(mesh_file, device=device)
            hSource_list.append(hSource)
    elif args.method == 'ihoi':
        mesh_list = sorted(glob(osp.join(exp_dir, 'meshes', '*_hObj.obj')))
        for mesh_file in mesh_list:
            hSource = mesh_utils.load_mesh(mesh_file, device=device)
            hSource_list.append(hSource)
        
    return hSource_list, it



@torch.no_grad()
def eval_mesh_in_hand(exp_dir, index):
    oTarget = get_gt_mesh(index)
    hTo_list, hA_gt_list = get_gt_hTo(index)
    
    # jSource --> hSource --> scale 0.1
    hSource_list, it = get_pred_hSource(exp_dir)

    hand_wrapper = hand_utils.ManopthWrapper().to(device)
    metric_list = []
    print(len(hSource_list), len(hTo_list))
    for i, (hSource, hTo_target, hA_gt) in enumerate(zip(hSource_list, hTo_list, hA_gt_list)):
        # hSource = mesh_utils.apply_transform(jSources, hTj_source)
        hTarget = mesh_utils.apply_transform(oTarget, hTo_target)

        _, _, tTs = icp_tool.register_meshes(hSource, hTarget, scale=True, N=1, seed=123, return_T=True, w=1)
        source_points = mesh_utils.ops_3d.sample_points_from_meshes(hSource, 10000)
        tSourcpoints = mesh_utils.apply_transform(source_points, tTs)

        # mean distance
        dist = ((tSourcpoints - source_points) ** 2).sum(-1).sqrt()

        dist = dist.mean() * 1000
        dist = dist.item()
        print('dist', dist)
        # if args.debug:
        #     hHand_source, _ = hand_wrapper(None, hA_pred)
        #     hHand_target, _ = hand_wrapper(None, hA_gt)
        #     hScene = mesh_utils.join_scene_w_labels([
        #         mesh_utils.join_scene([hHand_source, hSource]),
        #         mesh_utils.join_scene([hHand_target, hTarget]),
        #     ], 3,)
        #     image_list = mesh_utils.render_geom_rot(hScene, scale_geom=True)
        #     image_utils.save_gif(image_list, osp.join(exp_dir, 'align_hand', f'iter{it}_{i:03d}'))

        th_list = np.array([20]) * 1e-3
        f_list = mesh_utils.fscore(hSource, hTarget, th=th_list)
        f_list = np.array(f_list).reshape(-1)
        # cd is sum of squared chamfer distance
        f_list[-1] *= 1e4
        f_list = np.concatenate([f_list, [dist]])

        metric_list.append(f_list)
    f_list = np.mean(metric_list, axis=0)
    # convert list of list to str for pretty print in one line 
    f_str = ' '.join([f'{f:.3f}' for f in f_list])
    rtn = f'iter: {it} ' + f_str
    print(osp.basename(exp_dir), rtn)
    return rtn

@torch.no_grad()
def eval_mesh(exp_dir, index):
    mesh_file, it = get_pred_meshes(exp_dir)
    
    if isinstance(mesh_file, list):
        mesh_file_list = mesh_file
    else:
        mesh_file_list = [mesh_file]
    
    targets = get_gt_mesh(index)

    f_mean_list = []
    for mesh_file in mesh_file_list:
        sources = mesh_utils.load_mesh(mesh_file, device=device,scale_verts=0.1)
        new_s, _ = icp_tool.register_meshes(sources, targets, scale=args.scale, seed=123, N=10)
        th_list = np.array([5, 10, 20]) * 1e-3
        f_list = mesh_utils.fscore(new_s, targets, th=th_list)
        f_list = np.array(f_list).reshape(-1)
        # cd is sum of squared chamfer distance
        f_list[-1] *= 1e4
        f_mean_list.append(f_list)
    f_list = np.mean(f_mean_list, axis=0)

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
    elif args.method == 'ihoi':
        exp_list = sorted(glob(osp.join(args.dir, '*', 'meshes', )))
        exp_list = [osp.dirname(e) for e in exp_list]
    return exp_list

def get_index_list(exp_list):
    if args.data == 'hoi4d':
        rtn_list = index_list
    elif args.data == 'hhor':
        rtn_list = [osp.basename(e) for e in exp_list]
    return rtn_list


def one_cell():
    exp_list = [args.dir]
    index_list = [args.index]
    N = 10
    cell_list = [['' for _ in range(N)] for _ in range(len(index_list)+1)]
    name = f'{args.type}'
    if args.type == 'progress':
        func = get_iter
    elif args.type == 'eval':
        func = eval_mesh
        name += f'_scale{args.scale}'
    elif args.type == 'eval_in_hand':
        func = eval_mesh_in_hand
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


def make_fig(exp_dir, index, suf_list, nth):
    rtn_list = []
    for suf in suf_list:
        query = osp.join(exp_dir, 'vis_clip', f'*{suf}.png')
        image_list = sorted(glob(query))
        if len(image_list) == 0:
            query = osp.join(exp_dir, 'vis_clip', f'*{suf}*.png')
            image_list = sorted(glob(query))        
            
        if len(image_list) == 0:
            img = ''
        else:
            img = image_list[min(nth, len(image_list)-1)]
        rtn_list.append(img)

    return rtn_list

def all_fig():
    exp_list = get_exp_list()
    expname_list = [osp.basename(e) for e in exp_list]
    if args.method =='hhor':
        exp_list = [osp.join(e, 'handscanner') for e in exp_list]
    N = 10 # int(np.ceil(len(exp_list) / len(index_list)))
    print(exp_list)
    # create 2D list cell_list in shape of (len(index_list), N)
    suf_list = ['_gt', ]
    for d in args.degree_list.split(','):
        suf_list += [f'_{d}_hoi']
    for d in args.degree_list.split(','):
        suf_list += [f'_{d}_obj']
    # '_0_hoi', '_60_hoi', '_90_hoi', '_0_obj', '_60_obj', '_90_obj']
    cell_list = []
    
    # add index_list to the first col
    func = make_fig

    # for i, index in enumerate(tqdm(index_list, desc='instances')):
    for e, exp in enumerate(exp_list):
        rtn = func(exp, None, suf_list, args.nth)  # [inp, orig, 90, origin, 90]]
        cell_list.append([expname_list[e], ] + rtn)
            # for f, fig in enumerate(rtn):
            #     cell_list[f+1][m+1] = fig
            #     cell_list[f+1][0] = expname_list[m]
    # header 
    web_utils.run(osp.join(args.dir, 'vis', f'fig.html'), cell_list)
    


def main_fig():
    exp_list = get_exp_list()
    if args.method =='hhor':
        exp_list = [osp.join(e, 'handscanner') for e in exp_list]
    N = 10 # int(np.ceil(len(exp_list) / len(index_list)))
    index_list = get_index_list(exp_list)
    print(index_list, exp_list)
    # create 2D list cell_list in shape of (len(index_list), N)
    suf_list = ['_gt', ]
    for d in args.degree_list.split(','):
        suf_list += [f'_{d}_hoi']
    for d in args.degree_list.split(','):
        suf_list += [f'_{d}_obj']
    cell_list = [['' for _ in range(N)] for _ in range(len(index_list)*len(suf_list)+1)]
    # cell_list = []
    
    # add index_list to the first col
    for i, index in enumerate(index_list):
        cell_list[i+1][0] = index
    func = make_fig
    for i, index in enumerate(tqdm(index_list, desc='instances')):
        line = ['' for _ in range(N)]
        # filter exp_list that contains index
        exp_list_i = [e for e in exp_list if index in e]
        if len(exp_list_i) > N:
            print(exp_list_i)

        for m, exp in enumerate(exp_list_i):
            rtn = func(exp, index, suf_list, 1)  # [inp, orig, 90, origin, 90]]
            for f, fig in enumerate(rtn):
                cell_list[i*len(suf_list)+f+1][m+1] = fig
    line = ['' for _ in range(N)]
    # header 
    web_utils.run(osp.join(args.dir, 'vis', f'fig.html'), cell_list)
    

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
    elif args.type == 'eval_in_hand':
        func = eval_mesh_in_hand
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
    parser.add_argument('--index', type=str, default='Mug_1')
    parser.add_argument('--method', type=str, default='vhoi')
    parser.add_argument('--th', type=int, default=10000000000)
    parser.add_argument('--nth', type=int, default=1)
    parser.add_argument('--degree_list', type=str, default='0,60,90')
    parser.add_argument('--type', type=str, default='progress')
    parser.add_argument('--data', type=str, default='hoi4d')
    parser.add_argument('--hoi4d_tab', action='store_true')
    parser.add_argument('--hoi4d_fig', action='store_true')
    parser.add_argument('--all_fig', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--scale', action='store_true')
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    
    if args.hoi4d_tab:
        main_tab()
    if args.all_fig:
        all_fig()
    if args.hoi4d_fig:
        main_fig()
    if args.debug:
        one_cell()
    # if args.save_jHand:
        # save_hHand_t()
