import json
import os
import os.path as osp
import numpy as np
import argparse
import imageio
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from jutils import image_utils, hand_utils, mesh_utils, geom_utils
from pytorch3d.renderer import PerspectiveCameras
import torchvision.transforms.functional as TF
from .overlay_mocap import overlay_one


device = 'cuda:0'


def call_frank_mocap(box_dir, out_dir, **kwargs):
    """python -m demo.demo_handmocap --input_path /your/bbox_dir --out_dir ./mocap_output"""
    # [minX,minY,width,height]
    # {"image_path": "./sample_data/images/cj_dance_01_03_1_00075.png", "body_bbox_list": [[149, 380, 242, 565]], "hand_bbox_list": [{"left_hand": [288.9151611328125, 376.70184326171875, 39.796295166015625, 51.72357177734375], "right_hand": [234.97779846191406, 363.4115295410156, 50.28489685058594, 57.89691162109375]}]}
    frank_dir = '/home/yufeiy2/frankmocap/'
    cmd = f'cd {frank_dir}; \
          python -m demo.demo_handmocap --input_path {box_dir} --out_dir {out_dir} \
                    --view_type ego_centric --renderer_type pytorch3d --no_display \
                    --save_pred_pkl; cd -'
    print(cmd)
    os.system(cmd)
    return 


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def xxyy_to_xywh(hand_box):
    x1, y1, x2, y2 = hand_box
    return [x1, y1, x2-x1, y2-y1]

def save_hand_boxes(hand_paths, img_paths, hand_box_dir):
    def save_hand_box(hand_box, img_file, hand_file):
        # {"image_path": "xxx.jpg", 
        # "hand_bbox_list":[{"left_hand":[x,y,w,h], "right_hand":[x,y,w,h]}], 
        # "body_bbox_list":[[x,y,w,h]]}
        hand_box = xxyy_to_xywh(hand_box)
        out = {
            "image_path": img_file,
            "hand_bbox_list": [{"right_hand": hand_box}],
            "body_bbox_list": [hand_box]
        }
        json.dump(out, open(hand_file, 'w'), default=np_encoder)

    for hand_file, img_file in zip(hand_paths, img_paths):
        hand_mask = imageio.imread(hand_file)
        hand_box= image_utils.mask_to_bbox(hand_mask)
        
        index = osp.basename(img_file).split('.')[0]
        save_hand_box(hand_box, img_file, osp.join(hand_box_dir, f'{index}.json'))


def det_predicted_poses(data_dir , **kwargs):
    # Get all the images
    img_paths = sorted(glob(osp.join(data_dir, 'image/*.png')))

    # call frankmocap to get predicted poses
    call_frank_mocap(osp.join(data_dir, 'image'), data_dir, **kwargs)


def get_predicted_poses(data_dir):
    # Get all the images
    img_paths = sorted(glob(osp.join(data_dir, 'image/*.png')))
    # Get all the hand masks
    hand_paths = sorted(glob(osp.join(data_dir, 'hand_mask/*.png')))
    
    # save hand_boxes from hand_paths
    hand_box_dir = osp.join(data_dir, 'hand_boxes')
    os.makedirs(hand_box_dir, exist_ok=True)
    save_hand_boxes(hand_paths, img_paths, hand_box_dir)
    
    # call frankmocap to get predicted poses
    call_frank_mocap(hand_box_dir, data_dir)

    # post-preprocess the predicted poses
    post_process(data_dir, img_paths, )

    # # remove lock
    # os.rmdir(lock_file)


# def batch_ho3d(data_dir, ):
#     hand_paths = 
#     img_paths = 
#     ho3d_save_hand_boxes(hand_paths, img_paths, hand_box_dir)
#     call_frank_mocap(hand_box_dir, data_dir)
#     ho3d_post_process(data_dir, img_paths,)


def post_process(data_dir, img_paths):
    # 1. get into cameras_hoi.npz, hands.npz
    camera_dict = {'cTw': [], 'K_pix': []}
    hand_dict = {'hA': [], 'beta': []}
    # 2. vis the predicted poses
    out_paths = sorted(glob(osp.join(data_dir, 'mocap/*_prediction_result.pkl')))
    orig_H, orig_W = imageio.imread(img_paths[0]).shape[0:2]
    if osp.exists(osp.join(data_dir, 'cameras_hoi.npz')):
        fx = np.load(osp.join(data_dir, 'cameras_hoi.npz'))['K_pix'][0][0, 0]
        ndc_f = fx / orig_W * 2
    else:
        ndc_f = 1
    print(ndc_f)
    wrapper = hand_utils.ManopthWrapper().to(device)
    prev_data = None
    for img_file, out_file in zip(img_paths, out_paths):
        save_file = osp.join(data_dir, 'vis', osp.basename(img_file)[:-4])
        image, data = overlay_one(img_file, out_file, save_file, wrapper, render=False, f=ndc_f)
        if data is None:
            data = prev_data
            print('use previous', img_file)


        camera_dict['cTw'].append(data['cTh'].cpu().detach().numpy()[0])
        K_pix = mesh_utils.intr_from_ndc_to_screen(
            mesh_utils.get_k_from_fp(data['cam_f'], data['cam_p']), orig_H, orig_W)
        camera_dict['K_pix'].append(K_pix.cpu().detach().numpy()[0])
        hand_dict['hA'].append(data['hA'].cpu().detach().numpy()[0])
        hand_dict['beta'].append(data['betas'].cpu().detach().numpy()[0])
        prev_data = data

    np.savez_compressed(osp.join(data_dir, 'cameras_hoi_pred.npz'), **camera_dict)
    np.savez_compressed(osp.join(data_dir, 'hands_pred.npz'), **hand_dict)

    vis_K_pred(camera_dict, hand_dict, img_paths, data_dir, wrapper)
    # (3. remove folder)
    return 


def vis_K_pred(camera_dict, hand_dict, img_paths, data_dir, hand_wrapper, suf='pred'):
    for i, img_file in enumerate(tqdm(img_paths)):
        cTw = torch.FloatTensor(camera_dict['cTw'][i])[None].to(device)
        K_pix = torch.FloatTensor(camera_dict['K_pix'][i])[None].to(device)
        hA = torch.FloatTensor(hand_dict['hA'][i])[None].to(device)
        beta = torch.FloatTensor(hand_dict['beta'][i])[None].to(device)

        img = imageio.imread(img_file)
        H, W = img.shape[0:2]

        f, p = mesh_utils.get_fxfy_pxpy(mesh_utils.intr_from_screen_to_ndc(K_pix, H, H))

        cHand, cJoints = hand_wrapper(cTw, hA, th_betas=beta)
        cHand.textures = mesh_utils.pad_texture(cHand, 'blue')
        cameras = PerspectiveCameras(f, p).to(device)
        cHoi = cHand
        iHand = mesh_utils.render_mesh(cHoi, cameras, out_size=H)
        
        image_utils.save_images(iHand['image'], osp.join(data_dir, 'vis', f'{i:05d}_{suf}'),
                                bg=TF.to_tensor(img)[None], mask=iHand['mask'],)
    
    imageio.mimsave(osp.join(data_dir, f'overlay_{suf}.gif'),
                    [imageio.imread(e) for e in sorted(glob(osp.join(data_dir, f'vis/*_{suf}.png')))])
    return 

def smooth_hand(data_dir, args):
    cameras = np.load(osp.join(data_dir, 'cameras_hoi_pred.npz'))
    hands = np.load(osp.join(data_dir, 'hands_pred.npz'))
    cTw = torch.FloatTensor(cameras['cTw']).to(device)
    hA = torch.FloatTensor(hands['hA']).to(device)

    w_smooth = args.w_smooth
    # smooth the outliers of cTw by optimization 
    rot, trans, scale = geom_utils.homo_to_rt(cTw)
    rot = geom_utils.matrix_to_rotation_6d(rot)
    delta_rot = nn.Parameter(torch.zeros_like(rot))
    delta_trans = torch.zeros_like(trans)
    delta_scale = torch.zeros_like(scale)

    delta_hA = nn.Parameter(torch.zeros_like(hA))
    optimizer = torch.optim.AdamW([delta_rot, delta_hA], lr=1e-1)
    cTw = geom_utils.matrix_to_se3(cTw)

    hand_wrapper = hand_utils.ManopthWrapper().to(device)
    losses = {}

    def get_loss(cTw, hA, delta_cTw, delta_hA):
        # watch for this space~
        new_cTw = cTw + delta_cTw
        rot, trans, scale = torch.split(new_cTw, [6, 3, 3], dim=-1)
        new_cTw = torch.cat([rot, torch.zeros_like(trans), torch.ones_like(scale)], -1)
        
        cHand, cJoints = hand_wrapper(new_cTw, hA + delta_hA)
        # temporal smoothness of cJoints
        loss_smooth = torch.mean(torch.norm(cJoints[1:] - cJoints[:-1], dim=-1, p=2))
        # delta regularization
        loss_delta = torch.mean(torch.norm(delta_cTw, dim=-1, p=1)) + torch.mean(torch.norm(delta_hA, dim=-1, p=2))

        loss = w_smooth*loss_smooth + loss_delta
        losses['smooth'] = w_smooth * loss_smooth.item()
        # losses['lsos_verts'] = w_smooth * lsos_verts.item()
        losses['delta'] = loss_delta.item()
        return loss
    
    def closure():
        optimizer.zero_grad()
        delta_cTw = torch.cat([delta_rot, delta_trans, delta_scale], dim=-1)
        loss = get_loss(cTw, hA, delta_cTw, delta_hA)

        loss.backward()
        return loss
    
    for i in range(1000):
        optimizer.step(closure)
        # optimizer.step(closure)
        if i % 10 == 0:
            print(f'iter {i}: {closure()}')
            for k,v in losses.items():
                print(f'\t{k}: {v}')

    delta_cTw = torch.cat([delta_rot, delta_trans, delta_scale], dim=-1)
    cTw_end = cTw + delta_cTw
    cTw_end = geom_utils.se3_to_matrix(cTw_end)
    hA_end = hA + delta_hA

    new_cameras = {'K_pix': cameras['K_pix'], 'cTw': cTw_end.cpu().detach().numpy()}
    new_hands = {'hA': hA_end.cpu().detach().numpy(), 'beta': hands['beta']}

    vis_K_pred(new_cameras, new_hands, 
               sorted(glob(osp.join(data_dir, 'image/*.png'))), data_dir, hand_wrapper, suf=f'smooth_{args.w_smooth}')

    np.savez_compressed(osp.join(data_dir, f'cameras_hoi_smooth_{args.w_smooth}.npz'), **new_cameras)
    np.savez_compressed(osp.join(data_dir, f'hands_smooth_{args.w_smooth}.npz'), **new_hands)

def batch_overlay(data_dir):
    data_dirs = sorted(glob(osp.join(data_dir, '*/image')))
    for data_dir in data_dirs:
        data_dir = osp.dirname(data_dir)
        imageio.mimsave(osp.join(data_dir, 'overlay_pred.gif'),
                    [imageio.imread(e) for e in sorted(glob(osp.join(data_dir, 'vis/*.png')))])
        

def batch_get_predicted_poses(data_dir):
    data_dirs = sorted(glob(osp.join(data_dir, '*/image')))
    for inp_dir in tqdm(data_dirs, desc='batch_get_predicted_poses'):

        done = osp.join(osp.dirname(inp_dir), f'hands_smooth_{args.w_smooth}.npz')
        lock_file = done + '.lock'
        if args.skip and osp.exists(done):
            print(f'{done} exists, skip')
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if args.skip:
                print(f'lock {lock_file} exists, skip')
                continue

        get_predicted_poses(osp.dirname(inp_dir))
        smooth_hand(osp.dirname(inp_dir), args)

        os.rmdir(lock_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess hand prediction')
    parser.add_argument('--inp', type=str, default='/home/yufeiy2/scratch/result/HOI4D/')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ho3d', action='store_true')
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--w_smooth', type=float, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()        
    if args.batch:
        batch_get_predicted_poses(args.inp)
    if args.debug:
        get_predicted_poses(args.inp)
        smooth_hand(args.inp, args)
    if args.overlay:
        batch_overlay(args.inp)
    # if args.ho3d:
    #     batch_ho3d(args.inp)