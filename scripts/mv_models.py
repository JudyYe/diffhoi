from glob import glob
import os
import os.path as osp
import shutil

src_dir = '/private/home/yufeiy2/scratch/result/vhoi/reproduce'
dst_dir = '/private/home/yufeiy2/scratch/result/vhoi/release_reproduce'

cat_list = 'Mug,Bottle,Kettle,Bowl,Knife,ToyCar'.split(',')
ind_list = [1, 2]
index_list = [f'{cat}_{ind}' for cat in cat_list for ind in ind_list]

def mv_models():
    for index in index_list:
        dst_folder = osp.join(dst_dir, index)
        os.makedirs(dst_folder, exist_ok=True)
        src_folder = osp.join(src_dir, index)

        item_list = ['config.yaml', 'ckpts', 'vis_clip']
        for item in item_list:
            src_item = osp.join(src_folder, item)
            dst_item = osp.join(dst_folder, item)
            
            if osp.exists(dst_item):
                continue
            cmd = 'cp -r {} {}'.format(src_item, dst_item)
            print(cmd)
            os.system(cmd)


def pad_zero_images():
    data_dir = '/private/home/yufeiy2/scratch/result/RAW_WILD/bottle_2/'
    image_list = glob(osp.join(data_dir, 'images/*.png'))
    tmp_dir = osp.join(data_dir, 'images_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    for image in image_list:
        image_name = osp.basename(image)
        image_name = image_name.split('.')[0]
        image_name = image_name.zfill(4)
        dst_image = osp.join(tmp_dir, image_name+'.png')
        shutil.copy(image, dst_image)
    cmd = 'mv {} {}'.format(osp.join(data_dir, 'images'), tmp_dir+'_old')
    os.system(cmd)
    cmd = 'mv {} {}'.format(tmp_dir, osp.join(data_dir, 'images'))
    os.system(cmd)

pad_zero_images()


