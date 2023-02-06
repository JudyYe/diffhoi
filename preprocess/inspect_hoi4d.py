import imageio
import subprocess
import json
import os
import os.path as osp
import logging as log
from jutils import image_utils

H = 512
# hand_mask/  
# image/  
#   00000.png
# mask/  
# obj_mask/
# image.gif  
# cameras_hoi.npz  
# hands.npz  
data_dir = '/home/yufeiy2/scratch/data/HOI4D/'
exclude_list = ['rest', 'reachout', 'stop']
save_dir = '/home/yufeiy2/scratch/result/HOI4D/'

mapping = [
    '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
    'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
    'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]

# Bottle: C5

def get_one_clip(index, t_start, t_end):
    f_start = int(t_start*15)
    f_end = int(t_end*15)
    # get_image with object centric crops 

    for t in range(f_start, f_end):
        masks, bbox = read_masks(index, t)
        bbox_sq = image_utils.square_bbox(bbox, 0.5)

        mask_crop = image_utils.crop_resize(masks, bbox_sq, H)
        obj_mask = (mask_crop[..., 0] > 10) * 255    # red object, green hand
        hand_mask = (mask_crop[..., 1] > 10) * 255 

        imageio

    return


def continuous_clip(index):
    clips = []
    action_file = osp.join(data_dir, 'HOI4D_annotations/', index, 'action/color.json')
    with open(action_file) as fp:
        act = json.load(fp)
    start = 0 
    stop = 0
    record = False
    for ev in act['events']:
        if not record:
            if ev['event'].lower() in exclude_list:
                continue
            else:
                record = True
                start = ev['startTime']
        else:
            if ev['event'].lower() in exclude_list:
                record = False
                stop = ev['startTime']
                clips.append([start, stop])
    return clips


def decode_video(root, list_file):

    with open(osp.join(data_dir, 'Sets', list_file), 'r') as f:
        rgb_list = [os.path.join(root, i.strip(),'align_rgb') for i in f.readlines()]

    for rgb in rgb_list:
        depth = rgb.replace('align_rgb','align_depth')
        rgb_video = os.path.join(rgb, "image.mp4")
        # depth_video = os.path.join(depth, "depth_video.avi")

        cmd =  """ ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15 -qscale:v 2 {}/%05d.{} -loglevel quiet """.format(rgb_video, rgb, "jpg")

        print(cmd)
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            log.info(err.decode())

        # cmd = """ ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15 -qscale:v 2 {}/%05d.{} -loglevel quiet """.format(depth_video, depth, "png")
        # print(cmd)

        # p = subprocess.Popen(cmd, shell=True,
        #                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # out, err = p.communicate()
        # if err:
        #     log.info(err.decode())


def batch_clip():
    with open(osp.join(data_dir, 'Sets/test_vid_ins.txt')) as fp:
        index_list = [line.strip() for line in fp]
    for index in index_list:
        clips = continuous_clip(index)
        print(clips)

        for cc in clips:
            get_one_clip(index, cc[0], cc[1])
    return



if __name__ == '__main__':
    # decode_video('/home/yufeiy2/scratch/data/HOI4D/HOI4D_release/', 'test_vid_ins.txt')
    clips = continuous_clip('ZY20210800001/H1/C2/N46/S54/s02/T1')
    print(clips)