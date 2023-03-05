import pickle
import numpy as np
import os.path as osp
from glob import glob
import imageio
import argparse
import os
from preprocess.clip_pred_hand import call_frank_mocap, post_process, smooth_hand

data_dir = '/home/yufeiy2/scratch/result/VISOR/'
def pred_frankmocap(data_dir):
    # if not exist image/,  move frame_image/xxx.jpg to image/xxx.png
    if not os.path.exists(os.path.join(data_dir, 'image')):
        os.makedirs(os.path.join(data_dir, 'image'))
    image_file_list = glob(os.path.join(data_dir, 'frame_image', '*.jpg'))
    image_list = [imageio.imread(e) for e in image_file_list]
    for i, image in enumerate(image_list):
        index = osp.basename(image_file_list[i]).split('.')[0]
        imageio.imwrite(os.path.join(data_dir, 'image', f'{index}.png'), image)
    call_frank_mocap(osp.join(data_dir, 'image'), data_dir,)

    return 

def pose_process_mask(data_dir, img_paths):
    hand_dir = osp.join(data_dir, 'hand_mask')
    obj_dir = osp.join(data_dir, 'obj_mask')
    os.makedirs(hand_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)
    prev_hand = None
    for i, img_path in enumerate(img_paths):
        index = osp.basename(img_path).split('.')[0]
        hand_mask = imageio.imread(img_path.replace('image', 'frame_hand_mask'))
        obj_mask = imageio.imread(img_path.replace('image', 'frame_obj_mask'))

        fname = osp.join(data_dir, 'mocap', f'{index}_prediction_result.pkl')
        hand_list = pickle.load(open(fname, 'rb'))['hand_bbox_list']
        hand_bbox = hand_list[0]['right_hand']
        print(hand_bbox)
        xywh_to_xyxy = lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]]
        
        if hand_bbox is None:
            hand_bbox = prev_hand
            if hand_bbox is None:
                print(hand_list[0]['left_hand'])
                print(img_path)
                continue
        canvas = np.zeros_like(hand_mask, dtype=np.uint8)
        xy_hand_box = xywh_to_xyxy(hand_bbox)
        xy_hand_box = pad_box(xy_hand_box, 0.2, canvas.shape[1], canvas.shape[0])
        canvas[int(xy_hand_box[1]):int(xy_hand_box[3]), int(xy_hand_box[0]):int(xy_hand_box[2])] = 1
        hand_mask = hand_mask * canvas
        imageio.imwrite(osp.join(hand_dir, f'{index}.png'), hand_mask)
        imageio.imwrite(osp.join(obj_dir, f'{index}.png'), obj_mask)
        prev_hand = hand_bbox
    # make gif for hand_mask and obj_mask 
    hand_mask_list = sorted(glob(osp.join(hand_dir, '*.png')))
    obj_mask_list = sorted(glob(osp.join(obj_dir, '*.png')))
    image_list = []
    for hand_mask, obj_mask, img in zip(hand_mask_list, obj_mask_list, img_paths):
        index = osp.basename(hand_mask).split('.')[0]
        hand_mask = imageio.imread(hand_mask)
        obj_mask = imageio.imread(obj_mask)
        img = imageio.imread(img)
        canvas = np.zeros_like(img)
        canvas[:, :, 0] = hand_mask
        canvas[:, :, 1] = obj_mask
        canvas[:, :, 2] = img[:, :, 0]
        image_list.append(canvas)
    imageio.mimsave(osp.join(data_dir, 'overlay.gif'), image_list)



def pad_box(box, pad_ratio, width, height):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, x1 - w * pad_ratio)
    y1 = max(0, y1 - h * pad_ratio)
    x2 = min(width, x2 + w * pad_ratio)
    y2 = min(height, y2 + h * pad_ratio)
    return [x1, y1, x2, y2]


def post(data_dir):

    img_paths = sorted(glob(osp.join(data_dir, 'image', '*.png')))
    pose_process_mask(data_dir, img_paths)
    # post_process(data_dir, img_paths)



def batch_func(func):
    index_list = glob(osp.join(data_dir, '*_???'))
    for index in index_list:
        print('processing', index)
        func(index)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action='store_true')
    parser.add_argument("--pred", action='store_true')
    parser.add_argument("--smooth", action='store_true')
    parser.add_argument("--post", action='store_true')
    parser.add_argument("--index", )
    parser.add_argument("--w_smooth", default=100, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    print(args)

    func = None
    if args.pred:
        func = pred_frankmocap
    if args.post:
        func = post
    if args.smooth:
        func = lambda x: smooth_hand(x, args)
    
    batch_func(func)