import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import os.path as osp
from jutils import mesh_utils, web_utils
import tools.icp_recon as icp_tool
import torch

np.random.seed(123)
torch.manual_seed(123)

device = 'cuda:0'
data_dir = '/private/home/yufeiy2/scratch/data/HOI4D/'
result_dir = '/private/home/yufeiy2/scratch/result/HOI4D/'

cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
ind_list = [1,2]
index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]


def eval_one_mesh(pred_file, gt_file, scale=True):
    sources = mesh_utils.load_mesh(pred_file, device=device)
    targets = mesh_utils.load_mesh(gt_file, device=device)

    new_s, _ = icp_tool.register_meshes(sources, targets, scale=scale, seed=123, N=10)

    th_list = np.array([5, 10, 20]) * 1e-3
    f_list = mesh_utils.fscore(new_s, targets, th=th_list)
    f_list = np.array(f_list).reshape(-1)
    # cd is sum of squared chamfer distance
    f_list[-1] *= 1e4

    return f_list
#     f_mean_list.append(f_list)
# f_list = np.mean(f_mean_list, axis=0)

# def eval_one_index_hand(index, exp_dir):
#     exp_dir = osp.join(exp_dir, index)
#     hPred = mesh_utils.load_mesh(osp.join(exp_dir, 'hObj.obj'), device=device)

def collect_worst_tempaltes(N=-1):
    results = []
    for index in index_list:
        cat = index.split('_')[0]

        cad_dir = osp.join(data_dir, 'HOI4D_CAD_Model_for_release/rigid/{}/{:03d}.obj')
        set_file = osp.join(data_dir, 'Sets', 'all_contact_train_hand.csv')
        df = pd.read_csv(set_file)
        df = df[df['class'] == cat]

        train_index = df['vid_index'].values
        ins_list = [int(e.split('/')[3][1:]) for e in train_index]
        train_ins_list = sorted(list(set(ins_list)))

        f_mean_list = np.load(osp.join(result_dir, index, 'eval_template', f'trainset_N={N}.npy'))
        f_list = np.argmin(f_mean_list, axis=0)
        f_list[-1] = np.argmin(f_mean_list, axis=0)[-1]

        for i, f_ind in enumerate([0, 1, -1]):
            ind = f_list[f_ind]
            # print(ind, f_list, len(train_ins_list))
            # print(train_ins_list[ind])
            obj_file = cad_dir.format(cat, train_ins_list[ind])

            dst_file = osp.join(result_dir, index, f'worst_template/{i:03d}.obj')
            os.makedirs(osp.dirname(dst_file), exist_ok=True)
            os.system(f'cp {obj_file} {dst_file}')

    # results = np.array(results)


def eval_one_index(index, N=10, scale=True, pool='mean', reload=False):
    cat = index.split('_')[0]

    cad_dir = osp.join(data_dir, 'HOI4D_CAD_Model_for_release/rigid/{}/{:03d}.obj')
    set_file = osp.join(data_dir, 'Sets', 'all_contact_train_hand.csv')
    df = pd.read_csv(set_file)
    df = df[df['class'] == cat]

    train_index = df['vid_index'].values
    ins_list = [int(e.split('/')[3][1:]) for e in train_index]
    train_ins_list = sorted(list(set(ins_list)))

    # sample N from train_ins_list
    if N > 0:
        train_ins_list = np.random.choice(train_ins_list, N, replace=False)
    f_mean_list = []
    if not reload:
        for ins in tqdm(train_ins_list, desc=f'{index}'):
            pred_file = cad_dir.format(cat, ins)
            gt_file = osp.join(result_dir, index, 'oObj.obj')
            
            f_list = eval_one_mesh(pred_file, gt_file, scale=scale)
            f_mean_list.append(f_list)
    else:
        print('reload')
        f_mean_list = np.load(osp.join(result_dir, index, 'eval_template', f'trainset_N={N}.npy'))
    if pool == 'mean':
        f_list = np.mean(f_mean_list, axis=0)
    elif pool == 'worst':
        f_list = np.min(f_mean_list, axis=0)
        f_list[-1] = np.max(f_mean_list, axis=0)[-1]
    elif pool == 'best':
        f_list = np.max(f_mean_list, axis=0)
        f_list[-1] = np.min(f_mean_list, axis=0)[-1]
    elif pool == 'median':
        f_list = np.median(f_mean_list, axis=0)
    print(f_list)
    
    if not reload:
        f_mean_list = np.array(f_mean_list)  # (N, 4)
        title = ['F@5mm', 'F@10mm', 'F@20mm', 'CD']
        # plot 4 curves, each curve is a high to low order
        plt.figure(figsize=(10, 5*4))
        for i in range(f_mean_list.shape[1]):
            plt.subplot(4, 1, i+1)
            y = f_mean_list[:, i]
            y = np.sort(y)
            if i == f_mean_list.shape[1] - 1:
                y = y[::-1]
            plt.plot(y)
            plt.title(f'{title[i]}, mean={np.mean(y):.4f}')

        os.makedirs(osp.join(result_dir, index, 'eval_template'), exist_ok=True)
        plt.savefig(osp.join(result_dir, index, f'eval_template/trainset_N={N}.png'))
        np.save(osp.join(result_dir, index, f'eval_template/trainset_N={N}.npy'), f_mean_list)
        plt.close()

    f_str = ' '.join([f'{f:.3f}' for f in f_list])
    return f_str


def eval_all(N=10, **kwargs):
    results = []
    for index in index_list:
        res = eval_one_index(index, N=N, **kwargs)
        results.append(res)
    results = np.array(results)

    exp_list = ['/home/yufeiy2/scratch/result/homan/default_1_1/']
    exp_dir = '/home/yufeiy2/scratch/result/homan/default_1_1/'
    name = 'object'
    cell_list = [['' for _ in range(5)] for _ in range(len(index_list)+1)]
    # add index_list to the first col
    for i, index in enumerate(index_list):
        cell_list[i+1][0] = index

    for i, index in enumerate(tqdm(index_list, desc='instances')):
        # filter exp_list that contains index
        
        rtn = eval_one_index(index, N, **kwargs)
        cell_list[i+1][1] = f'{index:20}: ' + rtn

        # eval_one_index_hand(index, exp_dir)

    # pretty print cell_list
    for row in cell_list:
        print(' '.join([f'{c:10}' for c in row]))

    web_utils.run(osp.join(result_dir, index, f'eval_template/{name}_N={N}.html'), cell_list)

    return 


if __name__ == '__main__':
    # eval_one_index('Mug_1', N=-1)
    # eval_all(N=-1, pool='median', reload=True)
    collect_worst_tempaltes()