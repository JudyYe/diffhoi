import os.path as osp
from pytorch3d.structures import Meshes
import torch
from utils.hand_utils import ManopthWrapper
from jutils import image_utils, mesh_utils

vis_dir = '/glusterfs/yufeiy2/vhoi/vis'
device = 'cuda:0'
hA = torch.zeros([1, 45]).to(device)
hand_wrapper = ManopthWrapper().to(device)
hHand, _ = hand_wrapper(None, hA)
assert isinstance(hHand, Meshes)
textures = torch.ones_like(hHand.verts_padded())
for i in range(6):
    inds = getattr(hand_wrapper, 'contact_index_%d' % i)
    textures[:, inds, 0] = 0

hHand.textures = mesh_utils.pad_texture(hHand, textures)
# image_list = mesh_utils.render_geom_rot(hHand, scale_geom=True)
image_list = mesh_utils.render_geom_rot(hHand, 'el', scale_geom=True)
image_utils.save_gif(image_list, osp.join(vis_dir, 'contact'))