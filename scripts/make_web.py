import argparse
import os
from glob import glob
import sys
from nnutils import web_utils, slurm_utils

from utils import io_util

# compare

def main(args, config):
    log_dir = 'logs/'


    # exp_list = sys.argv[1].split(',')
    exp_list = args.exp.split(',')
    cell_list = [['gt_rgb', 'predicted_rgb', 'predicted mask', 'meshes']]
    for exp in exp_list:
        # last = lambda x: sorted(glob(os.path.join(log_dir, exp, x)))[-1]
        def last(x):
            try:
                y = sorted(glob(os.path.join(log_dir, exp, x)))[-1]
            except IndexError:
                y = x
                print(os.path.join(log_dir, exp, x))
            return y
        cell_list.append([last('imgs/val/gt_rgb/*.png'), 
                last('imgs/val/predicted_rgb/*.png'),
                last('imgs/val/pred_mask_volume/*.png'),
                last('meshes/*.ply'),
                ])
    web_utils.run(log_dir + '/cmp_exp/', 
        cell_list, width=400) 

if __name__ == '__main__':


    parser = io_util.create_args_parser()
    parser.add_argument("--exp", type=str, default=None, help='master port for multi processing. (if used)')

    slurm_utils.add_slurm_args(parser)
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)

    slurm_utils.slurm_wrapper(args, 'logs/test_cmp', main, {'args': args, 'config': config})
