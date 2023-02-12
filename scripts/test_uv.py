import os.path as osp
import torch
from pytorch3d.renderer.mesh.textures import TexturesUV
from pytorch3d.transforms import Translate
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.io import load_obj
from jutils import  mesh_utils, image_utils, geom_utils, hand_utils

# mesh_file = '/private/home/yufeiy2/vhoi/output/neurecon_out/open_hand_uv.obj'
device = 'cuda:0'
save_dir = '/home/yufeiy2/scratch/result/vis/'
hand_wrapper = hand_utils.ManopthWrapper().to(device)

H = 224
red, green = torch.meshgrid(
    torch.linspace(0, 1, H, device=device),
    torch.linspace(0, 1, H, device=device)
)  # (H, H)
tex = torch.stack([red, green, torch.zeros_like(red) - 1], -1)[None]  # (N, H, W, C)

N = 1 
hA = torch.zeros([N, 45], device=device)
hHand, _ = hand_wrapper(None, hA, texture=tex)
cHand, _ = mesh_utils.center_norm_geom(hHand)
f = 10
dist = f * 2
trans = Translate(torch.FloatTensor([[0, 0, dist]]), device=device)
cHand= mesh_utils.apply_transform(cHand, trans)
cameras = PerspectiveCameras(f).to(device)

image = mesh_utils.render_mesh(cHand, cameras, uv_mode=True)
image_utils.save_images(image['image'], osp.join(save_dir, 'image'))
uv = image['uv']
uv_color = torch.cat([uv, torch.ones_like(uv[:, 0:1])], 1)
image_utils.save_images(uv_color, osp.join(save_dir, 'uv'))


image_list = mesh_utils.render_geom_rot(cHand, 'el', scale_geom=True)
image_utils.save_gif(image_list, osp.join(save_dir, 'image'))


image_utils.save_images(tex.permute(0, 3, 1, 2), osp.join(save_dir, 'tex'))