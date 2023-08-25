"""Usage:  python tools/soft_link.py"""
import shutil
from glob import glob
import os
import os.path as osp 


exp2hoi4d_fig = {
    'ours': 'ablate_color/{}_rgb0',
    # 'ours_wild': 'wild/{}{}',
    # 'ihoi_wild': '../ihoi/light_mow/hoi4d/{}',

    'homan_avg': '../homan/other_1_1/{}_000',
    'homan_far': '../homan/worst_1_1/{}_000',

    'obj_prior': 'gray_which_prior_w0.01_exp/{}_suf_smooth_100_ObjGeomGlide_cond_all_linear_catTrue_cfgFalse',
    'hand_prior': 'gray_which_prior_w0.01_exp/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catFalse_cfgFalse',
    'no_prior': 'pred_no_prior_gray/{}_suf_smooth_100',

    'w_normal': 'ablate_weight_gray/{}_m1.0_n0_d1.0',
    'w_mask': 'ablate_weight_gray/{}_m0_n1.0_d1.0',
    'w_depth': 'ablate_weight_gray/{}_m1.0_n1.0_d0',

    'oTh': 'gray_oTh/{}_suf_smooth_100_False_False',
    'blend': 'oridnal_depth/{}_suf_smooth_100_depth1',

    'ihoi': '../ihoi/light_mow/hoi4d/{}',
    'hhor': '../hhor/hoi4d_go/{}',
    'gt': 'gt/{}',
}

method_list = ['homan_avg', 'homan_far']

# method_list = ['w_normal', 'w_mask', 'w_depth']
data_dir = '/private/home/yufeiy2/scratch/result/vhoi'
save_dir = '/private/home/yufeiy2/scratch/result/org'
def soft_link(method_list=None):
    if method_list is None:
        method_list = exp2hoi4d_fig.keys()
    for method in method_list:
        expname = exp2hoi4d_fig[method]
        seq_list = sorted(glob(osp.join(data_dir, expname.format('*'))))
        if len(seq_list) == 0:
            print('0', expname.format('*'))
        for seq_dir in seq_list:
            if not osp.isdir(seq_dir):
                print('not dir', seq_dir)
                continue
            seqname = osp.basename(seq_dir)
            suf = expname.format('*').split('*')[-1]
            if len(suf) > 0:
                seqname = seqname.split(suf)[0]
            dst_exp = osp.join(save_dir, method, seqname)

            # soft link from seq_dir to dst_exp
            os.makedirs(osp.dirname(dst_exp), exist_ok=True)
            if osp.exists(dst_exp) and osp.islink(dst_exp):
                cmd = 'rm %s' % dst_exp
                print(cmd)


            if not osp.exists(dst_exp):
                cmd = 'ln -s {} {}'.format(seq_dir, dst_exp)
                print(cmd)
                os.system(cmd)
soft_link(method_list)

exp2wild_fig = {
    'wild_ours': 'wild_gray/{}',
    'wild_ihoi': '../ihoi/light_mow/hoi4d/{}',
}


def get_seqname(seq_dir):
    seqname = osp.basename(seq_dir)
    potential = ['1st_nocrop', '3rd_nocrop', 'VISOR']
    for pot in potential:
        if pot in seqname:
            seqname = seqname.split(pot)[1]
            return seqname
    return seqname


def link_wild():
    for method, expname in exp2wild_fig.items():
        seq_list = sorted(glob(osp.join(data_dir, expname.format('*'))))
        for seq_dir in seq_list:
            if not osp.isdir(seq_dir):
                print('not dir', seq_dir)
                continue
            if osp.basename(seq_dir) == 'train':
                continue

            # seqname = osp.basename(seq_dir)
            seqname = get_seqname(seq_dir)
            if seqname is None:
                print(seq_dir)
                continue
            dst_exp = osp.join(save_dir, method, seqname)

            # soft link from seq_dir to dst_exp
            os.makedirs(osp.dirname(dst_exp), exist_ok=True)
            if osp.exists(dst_exp) and osp.islink(dst_exp):
                cmd = 'rm %s' % dst_exp
                print(cmd)

            if not osp.exists(dst_exp):
                cmd = 'ln -s {} {}'.format(seq_dir, dst_exp)
                print(cmd)
                os.system(cmd)


# link_wild()

