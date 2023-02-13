from PIL import Image
import numpy as np
import os
import os.path as osp
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from jutils import image_utils, geom_utils, mesh_utils


device = 'cuda:0'

def render_amodal_from_camera(hHand, hObj, cTh, cam_intr, H, W, depth=True,normal=True,uv=True):
    """_summary_

    :param hHand: _description_
    :param hObj: _description_
    :param cTh: _description_
    :param cam_intr: pixel space intr with size H, W
    :param H: _description_
    :param W: _description_
    :param depth: _description_, defaults to True
    :param normal: _description_, defaults to True
    :return: _description_
    """
    hHand = hHand.to(device)
    hObj = hObj.to(device)
    if cTh is not None:
        cTh = cTh.to(device)
        cHand = mesh_utils.apply_transform(hHand, cTh)
        cObj = mesh_utils.apply_transform(hObj, cTh)
    else:
        cHand = hHand
        cObj = hObj
    
    f, p = mesh_utils.get_fxfy_pxpy(mesh_utils.intr_from_screen_to_ndc(cam_intr, H, W))
    cameras = PerspectiveCameras(f, p).to(device)
    
    iHand = mesh_utils.render_mesh(cHand, cameras, 
        depth_mode=depth, 
        normal_mode=normal,
        uv_mode=uv,
        out_size=max(H, W))
    iHand['depth'] *= iHand['mask']
    iObj = mesh_utils.render_mesh(cObj, cameras, 
        depth_mode=depth, 
        normal_mode=normal,
        out_size=max(H, W))
    iObj['depth'] *= iObj['mask']

    # rescale to [0, 1] to be consistent with midas convention
    iHand['normal'] = iHand['normal'] / 2 + 0.5
    iObj['normal'] = iObj['normal'] / 2 + 0.5

    # sustract common min depth and scale to avoid overflow
    min_depth_hand = torch.min(torch.masked_select(iHand['depth'], iHand['mask'].bool()))
    min_depth_obj = torch.min(torch.masked_select(iObj['depth'], iObj['mask'].bool()))
    min_depth = min(min_depth_hand, min_depth_obj) - 2/1000
        # torch.masked_select(iHand['depth'], iHand['mask'].bool()).min(), 
        # torch.masked_select(iObj['depth'], iObj['mask'].bool()).min()) - 2/1000
    # print('norm', torch.norm(iHand['normal'], -1))
    iHand['depth'] *= 1000
    iObj['depth']  *= 1000   
    iHand['depth'] -= min_depth * 1000  # in mm
    iObj['depth'] -= min_depth * 1000
    return iHand, iObj


def save_amodal_to(iHand, iObj, save_index, cTh, hA, debug=False):
    image_utils.save_images(
        torch.cat([iHand['mask'], iObj['mask'], torch.zeros_like(iObj['mask']) ], 1), 
        save_index.format('amodal_mask'))

    save_depth(iHand['depth'], save_index.format('hand_depth'))
    save_depth(iObj['depth'], save_index.format('obj_depth'))

    os.makedirs(osp.dirname(save_index.format('hand_normal')), exist_ok=True)
    np.savez_compressed(save_index.format('hand_normal'), array=iHand['normal'][0].cpu().detach().numpy())  # (3, H, W)
    os.makedirs(osp.dirname(save_index.format('obj_normal')), exist_ok=True)
    np.savez_compressed(save_index.format('obj_normal'), array=iObj['normal'][0].cpu().detach().numpy())
    os.makedirs(osp.dirname(save_index.format('cTh')), exist_ok=True)
    os.makedirs(osp.dirname(save_index.format('hand_uv')), exist_ok=True)
    np.savez_compressed(save_index.format('hand_uv'), array=iHand['uv'][0].cpu().detach().numpy())  # (3, H, W)

    np.savez_compressed(save_index.format('cTh'), 
                        cTh=cTh[0].cpu().detach().numpy(),
                        hA=hA[0].cpu().detach().numpy())
    if debug:
        image_utils.save_images(iHand['normal'], save_index.format('hand_normal_vis'), scale=True)
        image_utils.save_images(iHand['normal'], save_index.format('obj_normal_vis'), scale=True)
    

def save_depth(image, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    image = image.cpu().detach()
    image = image[0].permute([1, 2, 0]).numpy()[..., 0]
    # image = np.clip(image, 0, 1<<8-1)
    # image = image.astype(np.uint8)
    image = np.clip(image, 0, 1<<16-1)
    image = image.astype(np.uint16)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    image = Image.fromarray(image)
    image.save(save_path + '.png')

