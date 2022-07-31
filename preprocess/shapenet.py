from glob import glob
import os
import os.path as osp

from tqdm import tqdm


src_dir = '/glusterfs/yufeiy2/download_data/ShapeNetPointCloud/ShapeNetCore.v2.PC15k/'
save_dir = '/glusterfs/yufeiy2/download_data/ShapeNetCore.v2/'

def make_list():
    for cls_file in tqdm(glob(osp.join(src_dir, '*'))):
        for split in ['train', 'val', 'test']:
            index_list = glob(osp.join(cls_file, split, '*.npy'))
            index_list = [osp.basename(e).split('.')[0] for e in index_list]
            synid = osp.basename(cls_file)
            print(len(index_list))
            with open(osp.join(save_dir, synid, '%s.txt' % split), 'w') as fp:
                for index in index_list:
                    meta_dir = osp.join(save_dir, '{}/{}/models/model_normalized.solid.binvox')
                    if osp.exists(meta_dir.format(synid, index)):
                        fp.write('%s\n' % index)
                    else:
                        print(osp.exists(meta_dir.format(synid, index)))
    
if __name__ == '__main__':
    make_list()