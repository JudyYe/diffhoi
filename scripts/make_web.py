import argparse
import os
import os.path as osp
from glob import glob
import sys
from jutils import web_utils, slurm_utils


# compare

def main(args):
    out_dir = args.out
    log_dir = args.log

    if args.exp is None:
        exp_list = [e.split('/')[-2] for e in glob(osp.join(log_dir, '*/ckpts'))]
    else:
        exp_list = args.exp.split(',')
    title_list = [
        'imgs/val/gt_rgb/*.png', 
        'imgs/val/predicted_rgb/*.png',
        'imgs/val/gt_flo_fw/*.png', 
        'imgs/val/predicted_flo_fw/*.png', 
        'imgs/val/pred_mask_volume/*.png',
        'imgs/val/pred_depth_volume/*.png',
        'meshes/*.ply',
        'cams/*.png',
    ]
    cell_list = [[' '] + [osp.basename(e.split('/*')[0]) for e in title_list], ]
    for exp in exp_list:
        # last = lambda x: sorted(glob(os.path.join(log_dir, exp, x)))[-1]
        def last(x):
            try:
                y = sorted(glob(os.path.join(log_dir, exp, x)))[-1]
            except IndexError:
                y = x
                print(os.path.join(log_dir, exp, x))
            return y
        line = [exp] + [last(e) for e in title_list]
        print(line)
        cell_list.append(line)
    web_utils.run(out_dir,
        cell_list, width=400) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None, help='master port for multi processing. (if used)')
    parser.add_argument("--log", type=str, default='../output/neurecon_out/', help='master port for multi processing. (if used)')
    parser.add_argument("--out", type=str, default='../output/neurecon_out/cmp_vis', help='')
    args, unknown = parser.parse_known_args()
    
    main(args)
