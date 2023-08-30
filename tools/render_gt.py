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
data_dir = '/private/home/yufeiy2/scratch/result/HOI4D'
save_dir = '/private/home/yufeiy2/scratch/result/vhoi/gt'
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

# def save_render(save_dir, t, data, out, H=512, W=512):


def save_render(save_dir, t, data, out, H=512, W=512):
    device = data['image'].device
    ww = hh = data['image'].size(-1)
    gt = data['image']

    degree_list = [0, 45, 60, 90, 180, 360-60, 360-90]
    name_list = ['gt', 'overlay_hoi', 'overlay_obj']
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
    # print(cHoi.verts_padded()[0, 0]    )
    iHoi = mesh_utils.render_mesh(cHoi, cameras, out_size=H,)
    # image1, mask1 = iHoi['image'], iHoi['mask']
    overlay = image_utils.blend_images(iHoi['image'], gt, iHoi['mask'],)
    image_list[0].append(gt*0.5+0.5)

    K_ndc = mesh_utils.get_k_from_fp(cam_f, cam_p)
    # print('out', cam_f, cam_p, cTh)
    hoi, obj = mesh_utils.render_hoi_obj_overlay(hHand, hObj, cTj=cTh, H=H, K_ndc=K_ndc)    
    # hoi_rgb, hoi_m = hoi.split([3, 1], 1)
    # hoi = hoi_rgb * hoi_m + gt * (1 - hoi_m)
    # image_list[0].append(gt)

    image_list[1].append(hoi)
    image_list[2].append(obj)
    # image_list[1].append(image_utils.blend_images(image1, gt, mask1))  # view 0

    for i, az in enumerate(degree_list):
        img1, img2 = mesh_utils.render_hoi_obj(hHand, hObj, az, cTj=cTh, H=H, W=W,)
        image_list[3 + 2*i].append(img1)  
        image_list[3 + 2*i+1].append(img2) 

    # save 
    for n, im_list in zip(name_list, image_list):
        im = im_list[-1]
        image_utils.save_images(im, osp.join(save_dir, f'{t:03d}_{n}'))

def render_video(data_list, video_dir):
    H = 512
    name_list = ['input', 'render_0', 'render_1', 'jHoi', 'jObj', 'vHoi', 'vObj', 'vObj_t', 'vHoi_fix']
    image_list = [[] for _ in name_list]

    for t, data in enumerate(tqdm(data_list, desc='frame')):
        data = model_utils.to_cuda(data, device)
        out = data
        jHand = out['hHand']
        jObj = out['hObj']
        cam_f = data['cam_f']
        cam_p = data['cam_p']
        cTh = data['cTh']
        cTh_mat = geom_utils.se3_to_matrix(cTh)

        K_ndc = mesh_utils.get_k_from_fp(cam_f, cam_p)
    
        image_list[0].append(data['image'] * 0.5 + 0.5)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, cTj=cTh_mat, H=H, K_ndc=K_ndc, bin_size=None)
        image_list[1].append(hoi)

        # rotate by 90 degree in world frame 
        # 1. 
        jTcp = mesh_utils.get_wTcp_in_camera_view(np.pi/2, cTw=cTh_mat)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTcp, H=H, K_ndc=K_ndc, bin_size=None)
        image_list[2].append(hoi)

        if t == (len(data_list)-1) // 2:
            # coord = plot_utils.create_coord(device, size=1)
            jHoi = mesh_utils.join_scene([jHand, jObj])
            image_list[3] = mesh_utils.render_geom_rot(jHoi, scale_geom=True, out_size=H, bin_size=0) 
            image_list[4] = mesh_utils.render_geom_rot(jObj, scale_geom=True, out_size=H, bin_size=0) 
            
            # rotation around z axis
            vTj = torch.FloatTensor(
                [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
                [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
            vHoi = mesh_utils.apply_transform(jHoi, vTj)
            vObj = mesh_utils.apply_transform(jObj, vTj)
            image_list[5] = mesh_utils.render_geom_rot(vHoi, scale_geom=True, out_size=H) 
            image_list[6] = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H) 

        jHoi = mesh_utils.join_scene([jHand, jObj])                
        vTj = torch.FloatTensor(
            [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
            [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
        vObj = mesh_utils.apply_transform(jObj, vTj)
        iObj_list = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H, bin_size=32) 
        image_list[7].append(iObj_list[t%len(iObj_list)])
        
        # HOI from fixed view point 
        scale = mesh_utils.Scale(5.0).to(device)
        trans = mesh_utils.Translate(0, 0.4, 0).to(device)
        fTj = scale.compose(trans)
        fHand = mesh_utils.apply_transform(jHand, fTj)
        fObj = mesh_utils.apply_transform(jObj, fTj)
        iHoi, iObj = mesh_utils.render_hoi_obj(fHand, fObj, 0, scale_geom=False, scale=1, bin_size=32)
        image_list[8].append(iHoi)    # save 
    for n, im_list in zip(name_list, image_list):
        for t, im in enumerate(im_list):
            image_utils.save_images(im, osp.join(video_dir, n, f'{t:03d}'))
    return 


def main():
    # index_list = ['Bowl_1']
    for vid in tqdm(index_list):
        render_dir = osp.join(save_dir, vid, 'vis_clip')
        video_dir = osp.join(save_dir, vid, 'vis_video')
        data_list = get_data(vid, data_dir)
        T = len(data_list) - 1
        render_step = np.linspace(0, T-1, T_num).astype(np.int).tolist() 
        # render_step = [0, T//2, T-1]
        render_video(data_list, video_dir, )
        # for t, data in enumerate(data_list):
        #     data = model_utils.to_cuda(data, device)

        #     if t in render_step:
        #         save_render(render_dir, t, data, data)
                # gt_render(render_dir, t, data, data)



if __name__ == '__main__':
    T_num = 10
    main()




    # vid = 
    # gt_render()