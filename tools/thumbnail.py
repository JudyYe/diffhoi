import numpy as np
import cv2
import imageio
from glob import glob
import os
import os.path as osp

ws_dir = '/private/home/yufeiy2/scratch/tmp'
inp_file = osp.join(ws_dir, 'bowl1_input.mp4')
out_file = osp.join(ws_dir, 'bowl1_wild_ours.mp4')
result_file = osp.join(ws_dir, 'result.gif')
H = 200

def encode_to_gif(image_list, out_file):
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    imageio.mimsave(out_file, image_list)


def extract_frames(video_file, out_dir):
    # extract by ffmpeg
    if osp.exists(osp.join(out_dir)):
        os.system('rm -rf {}/*'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    cmd = 'ffmpeg -i {} -r 10 -f image2 {}/%05d.jpg'.format(video_file, out_dir)
    os.system(cmd)
    
    # read those frames as image_list
    image_list = glob(osp.join(out_dir, '*.jpg'))
    image_list.sort()
    image_list = [imageio.imread(image_file) for image_file in image_list]
    return image_list

def make_video():
    inp_list = extract_frames(inp_file, osp.join(ws_dir, 'inp'))
    out_list = extract_frames(out_file, osp.join(ws_dir, 'out'))
    print(len(inp_list), len(out_list))
    min_t = min(len(inp_list), len(out_list))
    print('min_t', min_t)
    inp_list = inp_list[:min_t]
    out_list = out_list[:min_t]

    image_list = []

    for t, (inp_img, out_img) in enumerate(zip(inp_list, out_list)):
        inp_img = cv2.resize(inp_img, (H, H))
        # change black to white 
        # white_mask = np.all(inp_img < 20, axis=-1)
        # inp_img[white_mask] = 255

        # cut out_img to half width
        out_img = out_img[:, :out_img.shape[1]//2, :]
        out_img = cv2.resize(out_img, (H*2, H))


        result_img = np.concatenate([inp_img, out_img], axis=1)
        result_img[0:20] = 255
        result_img[H-25:H] = 255

        # result_img = result_img[20:H-25]

        margin = 50
        margin_img = np.ones((margin, result_img.shape[1], 3), dtype=np.uint8) * 255
        result_img = np.concatenate([margin_img, result_img, margin_img], axis=0)

        image_list.append(result_img)
    
    encode_to_gif(image_list, result_file)


make_video()