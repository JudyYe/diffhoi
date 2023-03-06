from tqdm import tqdm
from torchvision.transforms import ToTensor
from PIL import Image
from pytorch3d.renderer.cameras import PerspectiveCameras
import torch.nn.functional as F
from PIL import Image
from glob import glob
import numpy as np
import torch
import os
import os.path as osp
from jutils import mesh_utils, image_utils, hand_utils, geom_utils, model_utils

device = 'cuda:0'
cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
ind_list = [1,2]

index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]
data_dir = '/home/yufeiy2/scratch/result/HOI4D'
save_dir = '/home/yufeiy2/scratch/result/vhoi/gt'
hand_wrapper = hand_utils.ManopthWrapper().to(device)

def get_gt_mesh(index):
    mesh_dir = osp.join(data_dir, index, 'oObj.obj')
    mesh = mesh_utils.load_mesh(mesh_dir, device=device)

    return mesh

def get_gt_hTo(index):
    poses_dict = np.load(osp.join(data_dir, index, 'eval_poses.npz'))
    hTo_list = poses_dict['hTo']
    hA_gt = np.load(osp.join(data_dir, index, 'hands.npz'))['hA']
    hTo_list = [torch.FloatTensor(hTo).unsqueeze(0).to(device) for hTo in hTo_list]
    hA_list = [torch.FloatTensor(hA).unsqueeze(0).to(device) for hA in hA_gt]
    return hTo_list, hA_list


def get_data(vid, data_dir):
    hTo_list, hA_list = get_gt_hTo(vid)
    oObj = get_gt_mesh(vid)

    vid_dir = osp.join(data_dir, vid,)
    image_file_list = sorted(glob(osp.join(vid_dir, 'image', '*.*g')))
    data_list = []
    camera_dict = np.load(osp.join(vid_dir, 'cameras_hoi.npz'), )
    hand_dict = np.load(osp.join(vid_dir, 'hands.npz'), )
    for i, image_file in enumerate(image_file_list):
        data = {}
        image = Image.open(image_file)
        orig_W, orig_H = image.size
        image = image.resize((224, 224))
        mask = Image.open(image_file.replace('image', 'obj_mask')).resize((224, 224))
        data['image'] = ToTensor()(image).unsqueeze(0) * 2 - 1
        data['obj_mask'] = ToTensor()(mask).reshape(1, 1, 224, 224)
        cTh = torch.FloatTensor(camera_dict['cTw'][i])[None]
        data['cTh'] = geom_utils.matrix_to_se3(cTh)
        data['hA'] = torch.FloatTensor(hand_dict['hA'][i])[None]
        K_pix = torch.FloatTensor(camera_dict['K_pix'][i])[None]
        k_ndc = mesh_utils.intr_from_screen_to_ndc(K_pix, orig_H, orig_W)
        cam_f, cam_p = get_fp_from_k(k_ndc)
        data['cam_f'] = cam_f
        data['cam_p'] = cam_p
        data = model_utils.to_cuda(data, device)

        hTo = hTo_list[i]
        hA = hA_list[i]
        hHand, _ = hand_wrapper(None, hA.to(device))
        hObj = mesh_utils.apply_transform(oObj, hTo)
        data['hHand'] = hHand
        data['hObj'] = hObj
        
        cObj = mesh_utils.apply_transform(hObj, data['cTh'])
        cHand = mesh_utils.apply_transform(hHand, data['cTh'])
        data['cObj'] = cObj
        data['cHand'] = cHand

        data_list.append(data)

    return data_list


def get_fp_from_k(k_ndc):
    fx = k_ndc[:, 0, 0]
    fy = k_ndc[:, 1, 1]
    px = k_ndc[:, 0, 2]
    py = k_ndc[:, 1, 2]
    f = torch.stack([fx, fy], dim=-1)
    p = torch.stack([px, py], dim=-1)
    return f, p
def merge_hoi(jHand, jObj):
    jObj.textures = mesh_utils.pad_texture(jObj, 'white')
    jHand.textures = mesh_utils.pad_texture(jHand, 'blue')
    jMeshes = mesh_utils.join_scene([jHand, jObj])
    return jMeshes


def save_render(save_dir, t, data, out, H=512, W=512):
    device = data['image'].device
    ww = hh = data['image'].size(-1)
    gt = data['image']

    degree_list = [0, 45, 60, 90, 180, 360-60, 360-90]
    name_list = ['gt', 'overlay', ]
    for d in degree_list:
        name_list += ['%d_hoi' % d, '%d_obj' % d]  
    image_list = [[] for _ in name_list]

    hObj = out['hObj']
    hHand = out['hHand']
    cTh = data['cTh']
    cam_f = data['cam_f']
    cam_p = data['cam_p']

    gt = F.adaptive_avg_pool2d(gt, (H, W))
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    cHoi = merge_hoi(out['cHand'], out['cObj'])
    iHoi = mesh_utils.render_mesh(cHoi, cameras, out_size=H,)
    image1, mask1 = iHoi['image'], iHoi['mask']
    
    image_list[0].append(gt)
    image_list[1].append(image_utils.blend_images(image1, gt, mask1))  # view 0

    for i, az in enumerate(degree_list):
        img1, img2 = mesh_utils.render_hoi_obj(hHand, hObj, az, cTj=cTh, H=H, W=W)
        image_list[2 + 2*i].append(img1)  
        image_list[2 + 2*i+1].append(img2) 

    # save 
    for n, im_list in zip(name_list, image_list):
        im = im_list[-1]
        image_utils.save_images(im, osp.join(save_dir, f'{t:03d}_{n}'))


def main():
    for vid in tqdm(index_list):
        render_dir = osp.join(save_dir, vid, 'vis_clip')
        data_list = get_data(vid, data_dir)
        T = len(data_list) - 1
        render_step = [0, T//2, T-1]
        for t, data in enumerate(data_list):
            data = model_utils.to_cuda(data, device)
            if t in render_step:
                save_render(render_dir, t, data, data)


if __name__ == '__main__':
    main()