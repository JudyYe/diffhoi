import os
import os.path as osp
from sklearn.cluster import KMeans
import numpy as np
import pickle
from jutils import mesh_utils, image_utils, geom_utils
import torch
from utils.hand_utils import ManopthWrapper, get_nTh


K = 20
size = 1000
data_dir = '/glusterfs/yufeiy2/fair/data/obman/'
save_dir = data_dir
vis_dir = '/glusterfs/yufeiy2/vhoi/vis/'

def load_hA(split, size=None):
    index_list = [line.strip() for line in open(osp.join(data_dir, split + '.txt'))]

    hA_list = []
    for i, index in enumerate(index_list):
        anno = os.path.join(data_dir, split, 'meta_plus', index + '.pkl')
        with open(anno, 'rb') as fp:
            meta_info = pickle.load(fp)
        hA = meta_info['hA']
        hA_list.append(hA)
        if size is not None and i > size:
            break
    hA_list = np.array(hA_list)
    index_list = np.array(index_list)
    return hA_list, index_list

def write_subset(name, index_list):
    with open(osp.join(data_dir, name + '.txt'), 'w') as fp:
        [fp.write('%s\n' % e ) for e in index_list]


def main():
    device = 'cuda:0'
    hand_wrapper = ManopthWrapper().to(device)

    hA_list, index_list = load_hA('train', size)        

    kmeans = KMeans(n_clusters=K, random_state=0).fit(hA_list[:size])
    center = kmeans.cluster_centers_

    print(center.shape)  # (10, 45)
    _, hJoints = hand_wrapper(None, torch.FloatTensor(center).to(device))  # (N, J, 3)
    hJoints = hJoints.reshape(K, -1).cpu().detach().numpy()
    dist = np.sum((hJoints[:, None] - hJoints[None])**2, -1)  # 10, 10
    print(dist.shape)
    
    for i in range(K):
        for j in range(i+1):
            dist[i,j] = -1
    idx = np.argmin(-dist.reshape([-1]))
    print(dist)
    print(idx // K, idx % K, dist[idx//K, idx % K])

    mode_a = center[idx //K]
    mode_b = center[idx % K]

    center_ext = np.concatenate([mode_a[None], mode_b[None], center], 0)
    np.save(osp.join(save_dir, 'center%d.npy' % K), center_ext)

    hHand, _ = hand_wrapper(None, torch.FloatTensor([mode_a, mode_b]).to(device))
    image_utils.save_gif(mesh_utils.render_geom_rot(hHand, scale_geom=True), osp.join(vis_dir, 'hand'))

    hA_list, index_list = load_hA('train')
    label = kmeans.predict(hA_list)

    idx_a = np.where(label == idx //K )[0]
    idx_b = np.where(label == idx % K)[0]

    print(idx_a.shape, idx_b.shape, idx_a)
    idx = np.concatenate([idx_a, idx_b], -1).reshape([-1])
    np.random.shuffle(idx)

    print(len(idx), len(index_list))
    val_num = len(idx) // 10
    write_subset('train_mode', index_list[idx[val_num:]])
    write_subset('test_mode', index_list[idx[:val_num]])

    # hA_list, index_list = load_hA('test')
    # label = kmeans.predict(hA_list)

    # idx_a = np.where(label == idx //K )
    # idx_b = np.where(label == idx % K)

    # idx = np.stack([idx_a, idx_b], -1).reshape([-1])
    # if len(idx)

    # print(len(idx), len(index_list))
    # write_subset('test_mode', index_list[idx])


    # kmeans.labels_

    # kmeans.predict([[0, 0], [12, 3]])
    # kmeans.cluster_centers_


if __name__ == '__main__':
    np.random.seed(123)
    main()