import hydra
from glob import glob
import numpy as np
import imageio
import os
import os.path as osp
import argparse
import torch
from jutils import image_utils, geom_utils, mesh_utils, hand_utils, plot_utils
from preprocess.clip_pred_hand import post_process, get_predicted_poses, smooth_hand
cache_dir = '/home/yufeiy2/scratch/result/1st_cache'
odir = osp.join(cache_dir, 'by_obj')
pdir = osp.join(cache_dir, 'by_ppl')
fdir = osp.join(cache_dir, 'by_seq')
base_dir = '/home/yufeiy2/scratch/result/1st'

# image/  
#   00000.png
# obj_mask/
# hand_mask/  
# image.gif  

# cameras_hoi.npz  
# hands.npz  

# H, W = 480, 640
H, W = 512, 512
device = 'cuda:0'

def pad_to_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    elif h > w:
        pad = (h - w) // 2
        return np.pad(image, ((0, 0), (pad, pad), (0, 0)), )
    else:
        pad = (w - h) // 2
        return np.pad(image, ((pad, pad), (0, 0), (0, 0)), )
    
def square_crop_box_list(bbox_list, method, pad=0.2):
    
    sq_bbox_list =  [image_utils.square_bbox(bbox, pad) for bbox in bbox_list]
    sq_bbox_list = np.array(sq_bbox_list)  # T, 4
    if method == 'constant':
        xmin = np.min(sq_bbox_list[:, 0])
        ymin = np.min(sq_bbox_list[:, 1])
        xmax = np.max(sq_bbox_list[:, 2])
        ymax = np.max(sq_bbox_list[:, 3])
        sq_bbox_list = np.array([[xmin, ymin, xmax, ymax]] * len(sq_bbox_list))

    return sq_bbox_list

def random_gt_mesh(save_dir):
    mesh = plot_utils.create_coord('cpu', 1)
    mesh_utils.dump_meshes([osp.join(save_dir, 'oObj')], mesh)

def put_text(save_dir):
    text = osp.basename(save_dir)
    if '_' in text:
        text = text.split('_')[0]
    else:
        text = text[:-1]
    print(text)
    with open(osp.join(save_dir, 'text.txt'), 'w') as f:
        f.write(text)

def convert_from_custom_to_our_no_crop(seqname):
    random_gt_mesh(save_dir)
    put_text(save_dir)
    print(osp.join(odir, 'JPEGImages', f'{seqname}_0', '*.*g'))
    image_file_list = sorted(glob(osp.join(odir, 'JPEGImages', f'{seqname}_0', '*.*g')))
    obj_file_list = sorted(glob(osp.join(odir, 'VidAnnotations', f'{seqname}_0', '*.*g')))
    hand_file_list = sorted(glob(osp.join(pdir, 'Tracks', f'{seqname}_0', '*.*g')))
    hand_boxes_list = sorted(glob(osp.join(fdir, 'hand_box', f'{seqname}', '*.json')))

    image_list = [imageio.imread(f) for f in image_file_list]
    obj_list = [imageio.imread(f) for f in obj_file_list]
    hand_list = [imageio.imread(f) for f in hand_file_list]

    os.makedirs(osp.join(save_dir, 'image'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'obj_mask'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'hand_mask'), exist_ok=True)

    orig_H, orig_W = image_list[0].shape[:2]
    # 4x4 intrinsics 
    K = np.array([
        [orig_W, 0, orig_W / 2, 0], 
        [0, orig_H, orig_H / 2, 0], 
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        ])
    
    full_frame_K_intr = torch.FloatTensor(K)
    crop_box_list = []
    for image, obj, hand in zip(image_list, obj_list, hand_list):
        # pad to square
        hand_box = image_utils.mask_to_bbox(hand)
        obj_box = image_utils.mask_to_bbox(obj)
        hoi_box = image_utils.joint_bbox(hand_box, obj_box)
        crop_box_list.append(hoi_box)
        
    K_list = []
    crop_box_list = square_crop_box_list(crop_box_list, 'constant')
    for i, (image, obj, hand, hoi_box) in enumerate(zip(image_list, obj_list, hand_list, crop_box_list)):
        image = image_utils.crop_resize(image, hoi_box, H)
        obj = image_utils.crop_resize(obj, hoi_box, H)
        hand = image_utils.crop_resize(hand, hoi_box, H)
        cam_intr = image_utils.crop_cam_intr(
            full_frame_K_intr, 
            torch.FloatTensor(hoi_box), (H, H))
        K_list.append(cam_intr.cpu().numpy())

        image_file = image_file_list[i]
        obj_file = obj_file_list[i]
        hand_file = hand_file_list[i]

        imageio.imwrite(osp.join(save_dir, 'image', osp.basename(image_file).replace('.jpg', '.png')), image)
        imageio.imwrite(osp.join(save_dir, 'obj_mask', osp.basename(obj_file)), obj)
        imageio.imwrite(osp.join(save_dir, 'hand_mask', osp.basename(hand_file)), hand)

    K_list = np.array(K_list)
    print(K_list.shape)
    np.savez_compressed(osp.join(save_dir, 'cameras_hoi.npz'), K_pix=K_list)
    get_predicted_poses(save_dir)
    smooth_hand(save_dir, None, w_smooth=100)
    return 

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seq', type=str, default='bowl1,kettle6,knife2,knife3,mug3,mug2,bottle6,kettle4,knife6')
    # parser.add_argument('--seq', type=str, default='bottle_1,mug_1,mug_2,kettle_4,kettl_2,knife_3')
    parser.add_argument('--seq', type=str, default='bottle_1,bottle_2,mug_3,mug_1,kettle_4,kettl_2,kettle_5,knife_3,bowl_2,bowl_4,bowl_1')
    
    # knife1,bowl2,mug1,bottle2
    parser.add_argument('--to_ours', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--skip', action='store_true')
    return parser.parse_args()

@hydra.main(config_path='.', config_name='inspect_custom')
def batch_convert(args):
    seq = args.seq
    if args.to_ours:
        convert_from_custom_to_our_no_crop(seq)
    if args.smooth:
        random_gt_mesh(osp.join(base_dir + '_nocrop', seq))
        put_text(osp.join(base_dir + '_nocrop', seq))
        # smooth_hand(osp.join(base_dir + '_nocrop', seq), None, w_smooth=100)


if __name__ == '__main__':
    # batch_convert()
    args = parse_args()

    for seq in args.seq.split(','):
        seqname = seq
        save_dir = osp.join(base_dir + '_nocrop', seqname)
        done_file = osp.join(save_dir + '_nocrop', 'done', seqname)
        lock_file = osp.join(save_dir + '_nocrop', 'lock', seqname)  
        if args.skip and osp.exists(done_file):
            continue
        if args.skip and osp.exists(lock_file):      
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if args.skip:
                continue

        convert_from_custom_to_our_no_crop(seq)

        os.makedirs(done_file, exist_ok=True)
        os.system(f'rm -rf {lock_file}')


    # for seq in args.seq.split(','):
    #     convert_from_custom_to_our_no_crop(seq)
    # if args.smooth:
    #     random_gt_mesh(osp.join(base_dir + '_nocrop', seq))
    #     put_text(osp.join(base_dir + '_nocrop', seq))
    #     # smooth_hand(osp.join(base_dir + '_nocrop', seq), None, w_smooth=100)