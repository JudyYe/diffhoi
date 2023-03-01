import PIL
import trimesh
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
from jutils import image_utils, geom_utils, mesh_utils, hand_utils, plot_utils

from .read_write_model import *

device = 'cuda:0'
shape_dir = '/home/yufeiy2/scratch/data/HHOR/CAD/Sculptures/3_Giuliano.ply'
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
def convert_from_hhor_to_our():
    index = sorted(glob(osp.join(data_dir, 'colmap/images', '*.jpg')))
    index_list = [osp.basename(i).split('.')[0] for i in index]
    hand_wrapper = hand_utils.ManopthWrapper(side='right').to(device)
    K_intr_list = get_K_pix(data_dir)
    print('K, index', len(K_intr_list), len(index_list))
    
    cameras_dict = collections.defaultdict(list)
    hand_dict = collections.defaultdict(list)

    for i, index in enumerate(tqdm(index_list)):

        # image_orig = imageio.imread(osp.join(data_dir, 'colmap/images', f'{index}.jpg'))
        # image_orig = image_orig[:, ::-1]  # flip x
        # image_orig = pad_to_square(image_orig, )

        # # save to image/xxx.png
        # os.makedirs(osp.join(save_dir, 'image'), exist_ok=True)
        # imageio.imwrite(osp.join(save_dir, 'image', f'{index}.png'), image_orig)

        image = imageio.imread(osp.join(data_dir, 'colmap/semantics', f'{index}.png'))
        image = image[:, ::-1]  # flip x
        # hand: (255, 0, 255) obj: (255, 255, 0)
        image = pad_to_square(image, )
        hand_mask = ((image[..., 2] > 122.5) * (image[..., 0] > 122.5) * 255).astype(np.uint8)  # blue and purple
        obj_mask = ((image[..., 1] > 122.5) * 255).astype(np.uint8)  # green and yellow        

        # save to hand_mask/xxx.png
        os.makedirs(osp.join(save_dir, 'hand_mask'), exist_ok=True)
        os.makedirs(osp.join(save_dir, 'obj_mask'), exist_ok=True)
        imageio.imwrite(osp.join(save_dir, 'hand_mask', f'{index}.png'), hand_mask)
        imageio.imwrite(osp.join(save_dir, 'obj_mask', f'{index}.png'), obj_mask)

        # cTw, hA, beta = get_hand_param(int(index), hand_wrapper, side='right')

        # cameras_dict['cTw'].append(cTw.detach().cpu().numpy()[0])
        # hand_dict['hA'].append(hA.detach().cpu().numpy()[0])
        # hand_dict['beta'].append(beta.detach().cpu().numpy()[0])

        # # visualize rendered hand
        # hHand, _ = hand_wrapper(None, hA, th_betas=beta)
        # cHand = mesh_utils.apply_transform(hHand, cTw)
        # K = torch.FloatTensor(K_intr_list[i])[None]
        # K = mesh_utils.intr_from_screen_to_ndc(K, max(H, W), max(H, W))
        # fxfy, pxpy = mesh_utils.get_fxfy_pxpy(K)
        # cameras = PerspectiveCameras(fxfy, pxpy, device=device)

        # bg = TF.to_tensor(image_orig)[None]
        # iHand = mesh_utils.render_mesh(cHand, cameras, out_size=max(H, W), )
        # image_utils.save_images(iHand['image'], osp.join(save_dir, 'vis', f'{index}_hand_mesh'),
        #                         bg=bg, mask=iHand['mask'])
        # # rotate hand
        # image_list = mesh_utils.render_geom_rot(hHand, scale_geom=True)
        # image_utils.save_gif(image_list, osp.join(save_dir, 'vis', f'{index}_hand'))

    # cameras_dict['cTw'] = np.array(cameras_dict['cTw'])
    # cameras_dict['K_pix'] = np.array(K_intr_list)

    # np.savez_compressed(osp.join(save_dir, 'cameras_hoi.npz'), **cameras_dict)
    # np.savez_compressed(osp.join(save_dir, 'hands.npz'), **hand_dict)


def move_gt_mesh():
    target = trimesh.load(shape_dir)
    target.vertices -= target.center_mass
    target.vertices /= target.vertices.max()
    target.vertices[..., 0] *= -1
    target.faces = target.faces[:, [0, 2, 1]]
    trimesh.exchange.export.export_mesh(target, osp.join(save_dir, 'oObj.obj'))


def make_subset_gif():
    image_list = sorted(glob(osp.join(save_dir, 'image', '*.png')))
    image_list = [imageio.imread(i) for i in image_list]

    for ratio in [0.02, 0.05, 0.1, 0.2]:
        num = int(len(image_list) * ratio)
        imageio.mimsave(osp.join(save_dir, f'image_{ratio:g}.gif'), image_list[0:num])
    

def vis_camera():
    cTw = np.load(osp.join(save_dir, 'cameras_hoi.npz'))['cTw']
    cTw = torch.FloatTensor(cTw).to(device)
    for ratio in [0.02, 0.05, 0.1, 0.2]:
        num = int(len(cTw) * ratio)
        cTw_subset = cTw[0:num]
        coord = plot_utils.create_coord(device, size=0.1)
        mesh_list = plot_utils.vis_cam(cTw=cTw_subset, size=0.08 if ratio == 0.02 else None)
        scene = mesh_utils.join_scene(mesh_list + [coord])
        image_list = mesh_utils.render_geom_rot(scene, scale_geom=True, out_size=512)
        image_utils.save_gif(image_list, osp.join(save_dir, f'vis_camera_{ratio:g}'))


def create_gif():
    hand_wrapper = hand_utils.ManopthWrapper(side='right').to(device)

    image_list = sorted(glob(osp.join(save_dir, 'image', '*.png')))
    hand_mask_list = sorted(glob(osp.join(save_dir, 'hand_mask', '*.png')))
    obj_mask_list = sorted(glob(osp.join(save_dir, 'obj_mask', '*.png')))

    camera_dict = np.load(osp.join(save_dir, 'cameras_hoi.npz'))
    hand_dict = np.load(osp.join(save_dir, 'hands.npz'))

    blend_list = []
    for i, (image, hand_mask, obj_mask) in enumerate(zip(image_list, hand_mask_list, obj_mask_list)):
        image = imageio.imread(image)
        hand_mask = imageio.imread(hand_mask)
        obj_mask = imageio.imread(obj_mask)
        
        image = cv2.resize(image, (512, 512))
        hand_mask = cv2.resize(hand_mask, (512, 512))
        obj_mask = cv2.resize(obj_mask, (512, 512))

        cTw = torch.FloatTensor(camera_dict['cTw'][i])[None].to(device)
        K = torch.FloatTensor(camera_dict['K_pix'][i])[None].to(device)
        hA = torch.FloatTensor(hand_dict['hA'][i])[None].to(device)
        beta = torch.FloatTensor(hand_dict['beta'][i])[None].to(device)

        hHand, _ = hand_wrapper(None, hA, th_betas=beta)
        cHand = mesh_utils.apply_transform(hHand, cTw)

        K = mesh_utils.intr_from_screen_to_ndc(K, max(H, W), max(H, W))
        fxfy, pxpy = mesh_utils.get_fxfy_pxpy(K)
        cameras = PerspectiveCameras(fxfy, pxpy, device=device)

        bg = TF.to_tensor(image)[None]
        iHand = mesh_utils.render_mesh(cHand, cameras, out_size=512, )
        
        mask = np.stack([obj_mask > 0, hand_mask > 0, np.zeros_like(hand_mask)], axis=-1)
        blend = mask * image + (1-mask) * 0.5*image
        blend_list.append(blend.clip(0, 255).astype(np.uint8))
        
        image_utils.save_images(iHand['image'], osp.join(save_dir, 'overlay', f'{i:06d}_hand_mesh'),
                                bg=bg, mask=iHand['mask'])
    imageio.mimsave(osp.join(save_dir, 'image.gif'), blend_list)
    make_gif(osp.join(save_dir, 'overlay/*.png'), osp.join(save_dir, 'overlay.gif'))
    return 

def make_gif(img_dir, save_file):
    image_list = sorted(glob(img_dir))
    image_list = [imageio.imread(image) for image in image_list]
    imageio.mimsave(save_file, image_list)


def calibrate_rt(Rh, Th):
    cali_a = torch.FloatTensor([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ]).to(Rh)

    Th[..., 0] *= -1
    return Rh, Th


def render_one(index):
    side = 'right'
    hand_wrapper = hand_utils.ManopthWrapper(side=side).to(device)
    cTw, hA, beta =  get_hand_param(index, hand_wrapper, side=side)

    hHand, _ = hand_wrapper(None, hA, th_betas=beta)
    cHand = mesh_utils.apply_transform(hHand, cTw)

    cameras = read_cameras_binary(osp.join(data_dir, 'colmap/cameras.bin'))
    k = index // 5 + 1
    K = fxfy_pxpy_to_K(cameras[k].params)
    K = square_K(K, H, W)
    K = torch.FloatTensor(K)[None]
    K = mesh_utils.intr_from_screen_to_ndc(K, max(H, W), max(H, W))

    fxfy, pxpy = mesh_utils.get_fxfy_pxpy(K)
    cams = PerspectiveCameras(fxfy, pxpy, device=device)

    bg = imageio.imread(osp.join(data_dir, 'colmap/images', f'{index:06d}.jpg'))
    if side == 'right':
        bg = bg[:, ::-1]
    bg = pad_to_square(bg).copy().astype(np.uint8)
    bg = cv2.resize(bg, (224, 224))
    bg = TF.to_tensor(bg).to(device)[None]

    iHand = mesh_utils.render_mesh(cHand, cams, out_size=224, )
    image_utils.save_images(iHand['image'], osp.join(save_dir, 'vis', f'{index}_hand_mesh_{side}'),
                            bg=bg, mask=iHand['mask'])


def get_hand_param(index, hand_wrapper, side='right'):
    smpl_data = json.load(open(osp.join(data_dir, 'output/smpl', f'{index:06d}.json')))[0]

    rot = np.array(smpl_data['Rh'])
    beta = torch.FloatTensor(smpl_data['shapes']).to(device)  # (1, 10)
    # print('before', rot)
    if side == 'right':
        rot[..., 1::3] *= -1
        rot[..., 2::3] *= -1
    # print('after', rot)
    Rh = torch.FloatTensor(cv2.Rodrigues(rot)[0])[None].to(device)  # (1, 3, 3?)
    Th = torch.FloatTensor(smpl_data['Th']).to(device)  # (1, 3)?
    if side == 'right':
        Rh, Th = calibrate_rt(Rh, Th)
    wTh = geom_utils.rt_to_homo(Rh, Th)

    poses = torch.FloatTensor(smpl_data['poses']).to(device)  # (1, 48??
    poses[:, 3:] = hand_wrapper.pca_to_pose(poses[:, 3:], True)

    cTw = wTh
    hA = poses[:, 3:]
    return cTw, hA, beta



def get_K_pix(data_dir, ):
    cameras = read_cameras_binary(osp.join(data_dir, 'colmap/cameras.bin'))
    K_pix = []
    for k, v in cameras.items():
        K = fxfy_pxpy_to_K(v.params)
        K = square_K(K, H, W)
        K_pix.append(K)
    # k: 1:378
    return K_pix

def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data

def K_to_fxfypxpy(K):
    fx, fy, px, py = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return fx, fy, px, py

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



def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n"
    "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")



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


def batch_convert_from_our_to_hhor(data_dir):
    hhor_dir = data_dir + '_HHOR'
    seq_list = sorted(glob(os.path.join(data_dir, '*/image')))
    seq_list = [osp.dirname(e) for e in seq_list]

    for inp_dir in seq_list:
        seq = osp.basename(inp_dir)
        done = osp.join(hhor_dir, 'done', seq)
        lock = osp.join(hhor_dir, 'lock', seq)

        if args.skip and osp.exists(done):
            print('skip', seq)
            continue
        try:
            os.makedirs(lock)
        except FileExistsError:
            if args.skip:
                print('lose lock', seq, lock)
                continue
        out_dir = osp.join(hhor_dir, seq, 'colmap')
        convert_from_our_to_hhor(inp_dir, out_dir, args.suf)

        os.makedirs(done, exist_ok=True)
        os.rmdir(lock)


def convert_from_our_to_hhor(data_dir, out_dir, suf=''):
    """_summary_

    :param data_dir: result/HOI4D/Mug_1_0
    :param out_dir: result/HOI4D_HHOR/Mug_1_0/colmap
    """

    # camdata: dict 1-index 1...345
    # Camera(id=1, model='PINHOLE', width=1920, height=1080, params=array([1296., 1296.,  960.,  540.]))

    # img_out: dict 1-index 1...345
    # Image(id=1, 
        # qvec=array([ 0.82646548,  0.40011911, -0.38176071,  0.10544315]), 
        # tvec=array([0.023, 0.045, 0.389]), 
        # camera_id=1, name='000000.jpg', 
        # xys=array([], shape=(0, 2), dtype=float64), 
        # point3D_ids=array([], dtype=int64))    
    # points3D_out 1...778? (MANO vertices)
    # Point3D(id=1, 
    # xyz=array([-0.043396  , -0.01018524,  0.02160645], dtype=float32), 
    # rgb=array([0, 0, 0]), error=0, image_ids=array([0]), point2D_idxs=array([0]))

    image_list = sorted(glob(osp.join(data_dir, 'image', '*.png')))
    hand_list = sorted(glob(osp.join(data_dir, 'hand_mask', '*.png')))
    obj_list = sorted(glob(osp.join(data_dir, 'obj_mask', '*.png')))

    # save to colmap/images_png
    os.makedirs(osp.join(out_dir, 'images_png'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'semantics'), exist_ok=True)
    if not args.skip_image:
        for i, (image, hand, obj) in enumerate(zip(image_list, hand_list, obj_list)):
            index = osp.basename(image).replace('.png', '')
            out_image = osp.join(out_dir, 'images_png', '%s.png' % index)
            out_mask = osp.join(out_dir, 'semantics', '%s.png' % index)

            img = imageio.imread(image)
            hand_mask = imageio.imread(hand) > 0
            obj_mask = imageio.imread(obj) > 0
            mask = (hand_mask + obj_mask) > 0
            mask = mask.astype(np.uint8) * 255

            img_png = np.concatenate([img, mask[:, :, None]], axis=2)
            imageio.imwrite(out_image, img_png)

            hand_mask = hand_mask.astype(np.uint8) * 255
            obj_mask = obj_mask.astype(np.uint8) * 255
            sem_mask = np.stack([mask, obj_mask, hand_mask], axis=2)

            imageio.imwrite(out_mask, sem_mask)

    hand_wrapper = hand_utils.ManopthWrapper().to(device)
    # save to colmap/cameras.bin
    camera_out = {}
    camera_dict = np.load(osp.join(data_dir, f'cameras_hoi{suf}.npz'))
    hand_dict = np.load(osp.join(data_dir, f'hands{suf}.npz'))
    xys_ = np.zeros((0, 2), float)
    point3D_ids_ = np.full(0, -1, int)
    images_out = {}
    verts_list = []
    for i, (image, ) in enumerate(zip(image_list, )):
        # camera
        W, H = PIL.Image.open(image).size
        cTw = torch.FloatTensor(camera_dict['cTw'][i])

        K_pix = camera_dict['K_pix'][i]
        fxfypxpy = K_to_fxfypxpy(K_pix)
        camera_out[i+1] = Camera(id=i+1, model='PINHOLE', width=W, height=H, params=fxfypxpy)

        # posed image Rt
        imgname = osp.basename(image)
        Rnew, Tnew, s = geom_utils.homo_to_rt(cTw)  # cTw or wTc?? in form of K[Rt] (cTw)
        print(s, Tnew.shape)
        Rnew = Rnew.detach().cpu().numpy()
        Tnew = Tnew.detach().cpu().numpy()

        qvec = rotmat2qvec(Rnew)
        image_ = Image(
            id=i+1,
            qvec=qvec,
            tvec=Tnew,
            camera_id=i+1,
            name=imgname,
            xys=xys_,
            point3D_ids=point3D_ids_
        )
        images_out[i+1] = image_

    # points
    hA = torch.FloatTensor(hand_dict['hA']).to(device)
    beta = torch.FloatTensor(hand_dict['beta']).to(device)
    hHand, _ = hand_wrapper(None, hA, th_betas=beta)
    verts_all = hHand.verts_padded()
    points3D_out = {}
    vertices = verts_all.mean(0).cpu().detach().numpy()
    print(vertices.shape)
    for idx in tqdm(range(vertices.shape[0]), desc='points3D'):
        point_ = Point3D(
            id=idx+1,
            xyz=vertices[idx],
            rgb=np.array([0, 0, 0]),
            error=0,
            image_ids=np.array([0]),
            point2D_idxs=np.array([0])
        )
        points3D_out[idx+1] = point_

        
    write_cameras_binary(camera_out, os.path.join(out_dir, f"cameras{suf}.bin"))
    write_images_binary(images_out, os.path.join(out_dir, f"images{suf}.bin" ))
    write_points3d_binary(points3D_out, os.path.join(out_dir, f"points3D{suf}.bin"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default='')
    parser.add_argument('--suf', type=str, default='_smooth_100')
    parser.add_argument('--skip_image', action='store_true')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--skip', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # render_one(55)
    # convert_from_hhor_to_our()
    hoi4d_dir = '/home/yufeiy2/scratch/result/HOI4D'
    seq = args.seq
    if args.batch:
        batch_convert_from_our_to_hhor(hoi4d_dir)
    # convert_from_our_to_hhor(osp.join(hoi4d_dir, seq), osp.join(hoi4d_dir + '_HHOR', seq, 'colmap'), '_smooth_100')
    # create_gif()

    # move_gt_mesh()
    # make_subset_gif()
    # vis_camera()