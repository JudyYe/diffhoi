import os
import os.path as osp
import shutil

from tqdm import tqdm

src_dir = '/glusterfs/yufeiy2/fair/data/obman/'
dst_dir = '/scratch/yufeiy2/vhoi/obman/'

def cp_file(src_file, dst_file, mkdir=True):
    if mkdir:
        os.makedirs(osp.dirname(dst_file), exist_ok=True)
    # if osp.exists(dst_file):
    shutil.copyfile(src_file, dst_file, follow_symlinks=True)


def cp_meta():
    cp_file(osp.join(src_dir, 'train_mode.txt'), osp.join(dst_dir, 'train_mode.txt'))
    cp_file(osp.join(src_dir, 'test_mode.txt'), osp.join(dst_dir, 'test_mode.txt'))
    cp_file(osp.join(src_dir, 'center20.npy'), osp.join(dst_dir, 'center20.npy'))
    return 

def cp_grid():
    index_list = [line.strip() for line in open(osp.join(src_dir, 'train_mode.txt'))]
    index_list2 = [line.strip() for line in open(osp.join(src_dir, 'test_mode.txt'))]
    index_list += index_list2

    os.makedirs(osp.join(dst_dir, 'grid_sdf'), exist_ok=True)
    for index in tqdm(index_list):
        cp_file(osp.join(src_dir, 'grid_sdf/%s.npz' % index), osp.join(dst_dir, 'grid_sdf/%s.npz' % index), mkdir=False)
    return     


if __name__ == '__main__':
    cp_meta()
    # cp_grid()