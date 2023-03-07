"""Usage:  python tools/soft_link.py"""
import shutil
from glob import glob
import os
import os.path as osp 


exp2hoi4d_fig = {
    'ours': 'which_prior_w0.01_exp/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catTrue_cfgFalse',
    # 'ours_wild': 'wild/{}{}',
    # 'ihoi_wild': '../ihoi/light_mow/hoi4d/{}',

    'obj_prior': 'which_prior_w0.01_exp/{}_suf_smooth_100_ObjGeomGlide_cond_all_linear_catTrue_cfgFalse',
    'hand_prior': 'which_prior_w0.01_exp/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catFalse_cfgFalse',
    'no_prior': 'pred_no_prior/{}_suf_smooth_100',

    'w_normal': 'ablate_weight/{}_m1.0_n0_d1.0',
    'w_mask': 'ablate_weight/{}_m0_n1.0_d1.0',
    'w_depth': 'ablate_weight/{}_m1.0_n1.0_d0',
    'w_color': 'ablate_color/{}_rgb0',
    'anneal': 'which_prior_w0.01/{}_suf_smooth_100_CondGeomGlide_cond_all_linear_catTrue_cfgFalse',

    'ihoi': '../ihoi/light_mow/hoi4d/{}',
    'hhor': '../hhor/hoi4d_go/{}',
    'gt': 'gt/{}',

}


data_dir = '/home/yufeiy2/scratch/result/vhoi'
save_dir = '/home/yufeiy2/scratch/result/org'
def soft_link():
    for method, expname in exp2hoi4d_fig.items():
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

# soft_link()


exp2wild_fig = {
    'wild_ours': 'wild/{}',
    'wild_gray': 'wild_gray/{}',
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

link_wild()


exp2hhor_fig = {
    'hhor_ours':  
    {'AirPods':
        ['hhor_less_w0.001_1e-05/6_AirPods_40_0.05_suf',
         'hhor_less_w0.001_1e-05/6_AirPods_40_0.2_suf',
         'hhor_less_w0.001_1e-05/6_AirPods_40_0.5_suf'],
    'Doraemon':
         ['hhor_less_more_w0.001/28_Doraemon_40_0.05_suf',
          'hhor_less_more_w0.001/28_Doraemon_40_0.2_suf',
          'hhor_less_more_w0.001/28_Doraemon_40_0.9_suf'],
    },
    'hhor_hhor': {
    'AirPods':
        ['../hhor/less_data/6_AirPods_off40_0.05',
        '../hhor/less_data/6_AirPods_off40_0.2',
        '../hhor/less_data/6_AirPods_off40_0.5',],
    'Doraemon':
        ['../hhor/less_data_tiger/28_Doraemon_off40_0.05',
        '../hhor/less_data_tiger/28_Doraemon_off40_0.2',
        '../hhor/less_data_tiger/28_Doraemon_off40_0.9',],
    }

}

def link_hhor():
    for method, expgroup in exp2hhor_fig.items():
        for index, expname_list in expgroup.items():
            assert method not in exp2hoi4d_fig
            for i, expname in enumerate(expname_list):
                seqname = f'{index}_{i}'
                seq_dir = osp.join(data_dir, expname)

                dst_exp = osp.join(save_dir, method, seqname)

                # soft link from seq_dir to dst_exp
                os.makedirs(osp.dirname(dst_exp), exist_ok=True)
                if osp.exists(dst_exp) and osp.islink(dst_exp):
                    cmd = 'rm %s' % dst_exp
                    print(cmd)
                    os.system(cmd)

                if not osp.exists(dst_exp):
                    cmd = 'ln -s {} {}'.format(seq_dir, dst_exp)
                    print(cmd)
                    os.system(cmd)

# link_hhor()
