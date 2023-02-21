import collections
import struct
import cv2
from tqdm import tqdm
import pandas
from scipy.spatial.transform import Rotation as Rt
from glob import glob
import pickle
import numpy as np
import imageio
import subprocess
import json
import os
import os.path as osp
import logging as log
from PIL import Image
import argparse
import torch
import torchvision.transforms.functional as TF
from pytorch3d.renderer.cameras import PerspectiveCameras
from jutils import image_utils, geom_utils, mesh_utils, hand_utils



device = 'cuda:0'
data_dir = '/home/yufeiy2/scratch/data/HHOR/3_Giuliano/'
save_dir = '/home/yufeiy2/scratch/result/HHOR_volsdf/3_Giuliano/'
H = 1080 
W = 1920


# hand_mask/  
# image/  
#   00000.png
# mask/  
# obj_mask/
# image.gif  
# cameras_hoi.npz  
# hands.npz  
def convert():
    index = sorted(glob(osp.join(data_dir, 'colmap/images', '*.jpg')))
    index_list = [osp.basename(i).split('.')[0] for i in index]
    for index in tqdm(index_list):
        image_orig = imageio.imread(osp.join(data_dir, 'colmap/images', f'{index}.jpg'))
        image_orig = pad_to_square(image_orig, )

        # save to image/xxx.png
        os.makedirs(osp.join(save_dir, 'image'), exist_ok=True)
        imageio.imwrite(osp.join(save_dir, 'image', f'{index}.png'), image_orig)

        image = imageio.imread(osp.join(data_dir, 'colmap/semantics', f'{index}.png'))
        # hand: (255, 0, 255) obj: (255, 255, 0)
        image = pad_to_square(image, )
        hand_mask = ((image[..., 2] > 122.5) * (image[..., 0] > 122.5) * 255).astype(np.uint8)  # blue and purple
        obj_mask = ((image[..., 1] > 122.5) * 255).astype(np.uint8)  # green and yellow        

        # save to hand_mask/xxx.png
        os.makedirs(osp.join(save_dir, 'hand_mask'), exist_ok=True)
        os.makedirs(osp.join(save_dir, 'obj_mask'), exist_ok=True)
        imageio.imwrite(osp.join(save_dir, 'hand_mask', f'{index}.png'), hand_mask)
        imageio.imwrite(osp.join(save_dir, 'obj_mask', f'{index}.png'), obj_mask)

        # visualize masks
        vis_hand = 1.0 * image_orig * hand_mask[..., None] / 255
        vis_obj = 1.0 * image_orig * obj_mask[..., None] / 255
        vis_hand = vis_hand.clip(0, 255).astype(np.uint8)
        vis_obj = vis_obj.clip(0, 255).astype(np.uint8)
        os.makedirs(osp.join(save_dir, 'vis'), exist_ok=True)
        imageio.imwrite(osp.join(save_dir, 'vis', f'{index}_hand.png'), vis_hand)
        imageio.imwrite(osp.join(save_dir, 'vis', f'{index}_obj.png'), vis_obj)



def calibrate_rt(Rh, Th):
    joint = geom_utils.rt_to_homo(Rh, Th)
    cali_a = torch.FloatTensor([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ]).to(Rh)

    # Rh = cali_a @ Rh
    
    cali_b = torch.FloatTensor([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(Rh)

    # joint = cali_b @ joint
    # Rh, Th, _ = geom_utils.homo_to_rt(cali_b @ cali_a @ geom_utils.rt_to_homo(Rh, Th))
    Rh, Th, _ = geom_utils.homo_to_rt(joint)
    Th[..., 0] *= -1
    Rh = cali_a[..., 0:3, 0:3] @ Rh
    return Rh, Th

    hand_wrapper = hand_utils.ManopthWrapper(side=side).to(device)

def get_hand_param(index, hand_wrapper):
    # cameras = read_cameras_binary(osp.join(data_dir, 'colmap/cameras.bin'))

    side = 'right'
    bg = imageio.imread(osp.join(data_dir, 'colmap/images', f'{index:06d}.jpg'))
    bg = pad_to_square(bg).copy().astype(np.uint8)
    bg = cv2.resize(bg, (224, 224))
    bg = TF.to_tensor(bg).to(device)
    bg = bg[None]
    if side == 'right':
        bg = bg.flip([-1])

    smpl_data = json.load(open(osp.join(data_dir, 'output/smpl', f'{index:06d}.json')))[0]
    # hand_wrapper = hand_utils.ManopthWrapper(side='left').to(device)

    rot = np.array(smpl_data['Rh'])
    if side == 'right':
        rot[1::3] *= -1
        rot[2::3] *= -1
    Rh = torch.FloatTensor(cv2.Rodrigues(rot)[0])[None].to(device)  # (1, 3, 3?)
    Th = torch.FloatTensor(smpl_data['Th']).to(device)  # (1, 3)?
    if side == 'right':
        Rh, Th = calibrate_rt(Rh, Th)
    wTh = geom_utils.rt_to_homo(Rh, Th)

    poses = torch.FloatTensor(smpl_data['poses']).to(device)  # (1, 48??
    poses[:, 3:] = hand_wrapper.pca_to_pose(poses[:, 3:], True)

    cTw = wTh
    hA = poses[:, 3:]
    return cTw, hA


    hHand, _ = hand_wrapper(None, poses[:, 3:])
    wHand = mesh_utils.apply_transform(hHand, wTh)

    K = fxfy_pxpy_to_K(cameras[index].params)
    print(K)
    K = torch.FloatTensor(square_K(K, H, W))[None]
    print(K)
    print(K.shape)
    K = mesh_utils.intr_from_screen_to_ndc(K, max(H, W), max(H, W))
    fxfy, pxpy = mesh_utils.get_fxfy_pxpy(K)
    print(fxfy, pxpy)
    cameras = PerspectiveCameras(fxfy, pxpy, device=device)

    image = mesh_utils.render_mesh(wHand, cameras)
    image_utils.save_images(image['image'], osp.join(save_dir, 'vis', f'{index:06d}_hand_mesh_{side}'),
                            bg=bg, mask=image['mask'])
    # rotate hand
    image_list = mesh_utils.render_geom_rot(hHand, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, 'vis', f'{index:06d}_hand_{side}'))



def get_K_pix():
    cameras = read_cameras_binary(osp.join(data_dir, 'colmap/cameras.bin'))
    K_pix = []
    for k, v in cameras.items():
        print(k)
        K = fxfy_pxpy_to_K(v.params)
        K = square_K(K, H, W)
        K_pix.append(K)
    return K_pix

    # smpl_data = read_json(os.path.join(smpl_folder, smpl_name))[0]
    # vertices=body_model(return_verts=True, return_tensor=False, **smpl_data)[0]  # (778, 3)

    # # project
    # vert_3d = vertices.copy()
    # Rt = np.concatenate([R, T], 1)  # (3, 4)
    # vert_2d = np.concatenate([vert_3d, np.ones((vert_3d.shape[0], 1))], 1)
    # vert_2d = Rt @ vert_2d.transpose(1, 0)
    # vert_2d = K @ vert_2d
    # vert_2d = (vert_2d[:2, :] / (vert_2d[2:, :]+1e-5)).transpose(1, 0)
    # for point_2d in vert_2d:
    #     img_viz = cv2.circle(img, (int(point_2d[0]), int(point_2d[1])), radius=4, color=(0, 0, 255), thickness=-1)
    #     img_viz = cv2.resize(img_viz, (0, 0), fx=0.25, fy=0.25)
    # cv2.imshow('image', img_viz)
    # cv2.waitKey(0)
    # import ipdb; ipdb.set_trace()

    Rh = np.array(smpl_data['Rh'])
    Th = np.array(smpl_data['Th'])
    rot, _ = cv2.Rodrigues(Rh)
    Rtmano = np.concatenate((rot, Th[0][:, None]), -1)  # (3, 4)
    Rtmano = np.concatenate((Rtmano, [[0, 0, 0, 1]]), 0)

    smpl_data['Rh'] = np.zeros((1, 3))
    smpl_data['Th'] = np.zeros((1, 3))
    print(Rtmano.shape, smpl_data['poses'].shape, smpl_data['shapes'].shape)
    # vertices=body_model(return_verts=True, return_tensor=False, **smpl_data)[0]  # (778, 3)


def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data


def fxfy_pxpy_to_K(fx_fy_px_py):
    fx, fy, px, py = fx_fy_px_py
    K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    return K


def square_K(K, H, W):
    if H > W:
        # py = px
        K[..., 0, 2] = K[..., 1, 2]
    elif H < W:
        # px = py
        K[..., 1, 2] = K[..., 0, 2]
    return K

def pad_to_square(image, pad_value=0):
    h, w = image.shape[:2]
    if h == w:
        return image
    if h > w:
        image = np.pad(image, ((0, 0), ((h - w) // 2, (h - w) // 2), (0, 0)), 'constant', constant_values=pad_value)
    else:
        image = np.pad(image, (((w - h) // 2, (w - h) // 2), (0, 0), (0, 0)), 'constant', constant_values=pad_value)
    return image


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
            # (1296.0, 1296.0, 960.0, 540.0) PINHOLE
        assert len(cameras) == num_cameras
    return cameras



def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])

if __name__ == '__main__':
    render_one(25)
    # convert()
    