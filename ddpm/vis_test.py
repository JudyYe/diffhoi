import os
import os.path as osp
from glob import glob
from jutils import web_utils

def vis_eval_hand():
    base_dir = '/glusterfs/yufeiy2/vhoi/output_ddpm/art/'
    html_root = base_dir + 'vis_eval_hand'
    samp = [0, 1]
    # html_root = base_dir + 'vis_eval_hand_sample'
    # samp = [1]

    exp_list = glob(osp.join(base_dir, '*/eval_vary_hand'))
    T = [1, 100, 200, 500, 700,]
    image_list = [['index'] + ['T=%d/1000' % e for e in T]]
    for idx in range(8):
        for s in samp:
            for exp in exp_list:
                index = '%s idx %d qsample %d' % (exp.split('/')[-2], idx, s)
                row = [index]
                for t in T:
                    row.append(osp.join(exp, '%02d_hoi_T%d_%d_test_full.gif' % (idx, t, s)))
                image_list.append(row)
    web_utils.run(html_root, image_list, width=1000)




vis_eval_hand()