import json
import os
import os.path as osp
from pprint import pprint
import random
import numpy as np

import yaml

from .nb_video_utils import plot_frames_with_bboxes


data_dir = '/glusterfs/yufeiy2/download_data/EGO4D/ego4d_data/v1/'
vis_dir = '/glusterfs/yufeiy2/vhoi/vis_ego4d/'
def all_vid(split='val'):
    df = json.load(open(data_dir + '/annotations/fho_hands_%s.json' % split))
    video_uid = [e['video_uid'] for e in df['clips']]
    video_uid = list(set(video_uid))
    video_uid = sorted(video_uid)
    print(len(video_uid))
    return video_uid, df


def fho_tools(split='val'):
    df = json.load(open(data_dir + '/annotations/fho_scod_%s.json' % split))
    video_uid = [e['video_uid'] for e in df['clips']]
    video_uid = list(set(video_uid))
    print(len(video_uid))
    return video_uid, df

def copy(vid_list):
    for vid in vid_list:
        dst_file = '/glusterfs/yufeiy2/download_data/EGO4D/ego4d_data/v1/full_scale/%s.mp4' % vid
        if osp.exists(dst_file):
            print('Skip', dst_file)
            continue
        cmd = 'scp yufeiy2@grogu.ri.cmu.edu:/grogu/user/hmittal/ego4d/v1/full_scale/%s.mp4 /glusterfs/yufeiy2/download_data/EGO4D/ego4d_data/v1/full_scale/' % vid
        os.system(cmd)



def display(vid_list, scod_df, clip_id_list, hand_df=None, oscc_df=None):
    scod_df['clips'] = [e for e in scod_df['clips'] if e['video_uid'] in vid_list]
    hand_df['clips'] = [e for e in hand_df['clips'] if e['video_uid'] in vid_list]
    print(len(hand_df['clips']), len(scod_df['clips']))
    if clip_id_list[0] == -1:
        clip_id_list = np.random.permutation(range(len(scod_df['clips'])))
    # clips_df_list = np.random.permutation(scod_df['clips'])
    # print(clips_df_list)
    clips_df_list = scod_df['clips']
    for clip_id in clip_id_list:
        clip = clips_df_list[clip_id]
        
        vid = clip['video_uid']
        # Display critical frames for an action as a grid
        frame_order = ['pre_45', 'pre_30', 'pre_15', 'pre_frame', 'contact_frame', 'pnr_frame', 'post_frame']

        contact_frame = find_contact_frame(clip['clip_uid'], hand_df['clips'])
        frames.append(contact_frame)

        frames = [clip[e] for e in frame_order if e in clip]
        print(frames)
        fho_video_path = data_dir + '/full_scale/%s.mp4' % vid

        plot_frames_with_bboxes(fho_video_path, frames, fpath=os.path.join(vis_dir, vid + '_%d' % clip_id), key='bbox')    


def find_contact_frame(clip_uid, hand_clips):
    clip = [e for e in hand_clips if e['clip_uid'] == clip_uid][0]
    print(clip['frames'][0])
    print(clip['frames'][1])
    frame = {
        'bbox': [],
        'frame_number':  0
    }
    return frame


if __name__ == '__main__':
    import sys
    split = 'val'

    clip_id_list = [int(sys.argv[1])]

    vid_list, hand_df = all_vid(split)
    _ , scod_df = fho_tools(split)
    oscc_df = json.load(open(data_dir + '/annotations/' +'fho_oscc-pnr_%s.json' % split))

    copy(vid_list[0:10])
    display(vid_list[0:10], scod_df, clip_id_list, hand_df)



