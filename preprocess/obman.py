import os
import os.path as osp
import pickle

from tqdm import tqdm

data_dir = '/glusterfs/yufeiy2/download_data/obman/obman'
def list_3d():
    for split in ['val', 'test', 'train']:
        index_list = [index.strip() for index in open(osp.join(data_dir, '%s.txt' % split))]
        wr_fp = open(osp.join(data_dir, '%s_cad.txt' % split), 'w')        
        meta_dir = osp.join(data_dir, split, 'meta/{}.pkl')
        for index in tqdm(index_list):
            meta_file = meta_dir.format(index)
            with open(meta_file, 'rb')  as fp:
                meta_info = pickle.load(fp)
            wr_fp.write('%s %s %s\n' % (index, meta_info["class_id"], meta_info["sample_id"]))
        wr_fp.close()
    return 


if __name__ == '__main__':
    list_3d()
