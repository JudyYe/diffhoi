import os
import os.path as osp
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt


from dataio.DTU import SceneDataset

from jutils import geom_utils, mesh_utils

save_dir = '../output/neurecon_out/vis/'
os.makedirs(save_dir, exist_ok=True)
def main():
    # dataset = SceneDataset(False, './data/DTU/scan65')
    # mesh = mesh_utils.load_mesh('/checkpoint/yufeiy2/vhoi_out/neurecon_out/syn/flow_dtu/meshes/00021500.ply')

    dataset = SceneDataset(False, '/checkpoint/yufeiy2/vhoi_out/syn_data/00006755/')
    mesh = mesh_utils.load_mesh('/checkpoint/yufeiy2/vhoi_out/syn_data/00006755/gt.obj')


    loader = DataLoader(dataset)
    for i, data in enumerate(loader):
        if i > 1:
            break
    gt = data[2]
    data = data[1]
    # data = next(iter(dataset))[1]
    # data = next(loader)[1]

    # extrinsics = c2w = dataset.get_gt_pose(scaled=True).data.cpu().numpy()
    c2w = data['c2w'].detach().numpy().reshape([4, 4])
    cTw = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    wTc = c2w
    intr = data['intrinsics'].data.cpu().numpy()[0]
    image = gt['rgb'].reshape(dataset.H, dataset.W, 3).cpu().detach().numpy()

    verts = mesh.verts_packed().cpu().detach().numpy()  # (P, 3)

    # print(intr)
    proj_mat = intr @ cTw
    # proj_mat = dataset.proj_mat[i]
    uv = proj(verts, proj_mat)
    print('uv', proj_mat.shape, verts.shape, uv, uv.shape)

    draw(uv, image)
    plt.savefig(save_dir + '/look.png')

def draw(uv, image):
    plt.close()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.scatter(uv[..., 0], uv[..., 1])
    plt.subplot(1,2,2)
    plt.imshow(image)


def proj(verts, mat):
    ones = np.ones([len(verts), 1])
    homo_v = np.concatenate([verts, ones], -1)  # P, 4
    out_v = homo_v @ mat.T  # 4, 4
    print(homo_v.shape, mat.shape, out_v.shape)
    out = out_v[..., 0:2] / out_v[..., 2:3]
    return out

if __name__ == '__main__'    :
    main()