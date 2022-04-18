import pickle
import numpy as np
import scipy
from scipy.signal import butter, lfilter, freqz, lfilter_zi
from scipy.signal import savgol_filter

import os.path as osp
from matplotlib import pyplot  as plt
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.cameras import PerspectiveCameras
from models.cameras.para import PoseNet
from preprocess.inspect_100doh import vis_dataset, get_dataloader, vis_dir
from jutils import image_utils, mesh_utils, geom_utils
from utils.hand_utils import ManopthWrapper

seq = 'study_v_im0FA2X6fp0_frame000043_0'
device='cuda:0'


def vis_hA(hA_list, fname, title='', D_list=None):
    # list of [(45, ), ]
    if D_list is None:
        D_list = hA_list[0].shape[-1]
    D = len(D_list)

    T = len(hA_list)
    plt.close()
    fig = plt.figure(figsize=(6, D * 2))
    plt.title(title)
    for d in D_list:
        ax1 = fig.add_subplot(D, 1, d+1)
        # plt.subplot(D, 1, d+1)
        ax1.title.set_text('dim=%d'%d)
        v_list = [hA_list[t][d].item() for t in range(T)]
        ax1.plot(range(T), v_list)
        ax1.set_xticklabels(())

    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.savefig(fname + '.png')
    return


def vis_cTh(cTh_list):
    return


def vis_inp():
    dataloader = get_dataloader(seq)
    hA_list = []
    for data in dataloader:
        hA_list.append(data['hA'][0])
    vis_hA(hA_list, osp.join(vis_dir, 'inp'), 'input', range(6))

def optimize(wt=1, epoches=100, lr=1e-4, name='opt'):
    hand_wrapper = ManopthWrapper().to(device)
    dataloader = get_dataloader(seq, 8, False)
    T = len(dataloader.dataset)
    cTh = torch.stack(dataloader.dataset.cTh)
    print(cTh.shape)

    hA_delta = nn.Parameter(torch.zeros([T, 45]).to(device))
    cTh_net = PoseNet(T, True, True, init_c2w=cTh).to(device)
    params = list(cTh_net.parameters()) + [hA_delta]

    opt = optim.AdamW(params, lr)
    for ep in range(epoches):
        for data in dataloader:
            for k, v in data.items():
                try:
                    data[k] = v.to(device)
                except AttributeError:
                    pass

            opt.zero_grad()

            cameras = PerspectiveCameras(data['cam_f'], data['cam_p'], device=device)
            
            hA = hA_delta[data['inds']] + data['hA']
            cTh = cTh_net(data['inds'], None, None)

            hHand, _ = hand_wrapper(None, hA)
            cHand, cJoints = hand_wrapper(geom_utils.matrix_to_se3(cTh), hA)

            iHand = mesh_utils.render_soft(cHand, cameras)

            mask_loss = F.l1_loss(iHand['mask'], data['hand_mask'])
            temporal_loss = ((cJoints[1:] - cJoints[:-1])**2).mean()  # (T-1, J, 3)

            loss = mask_loss + wt * temporal_loss
            loss.backward()

            print('%.4f | %.4f | %.4f' % (loss, mask_loss, wt * temporal_loss))
            opt.step()
    dataloader = get_dataloader(seq, 1, False)
    
    name_list = ['gt', 'cHand', 'hHand', 'hHand_1']
    image_list = [[] for _ in name_list]
    hA_list, cTh_list = [], []
    for data in dataloader:
        for k, v in data.items():
            try:
                data[k] = v.to(device)
            except AttributeError:
                pass
        image_list[0].append(data['image'])

        hA = hA_delta[data['inds']] + data['hA']
        cTh = cTh_net(data['inds'], None, None)
        hA_list.append(hA[0].cpu())
        cTh_list.append(cTh[0].cpu())

        cameras = PerspectiveCameras(data['cam_f'], data['cam_p'], device=device)
        hHand, _ = hand_wrapper(None, hA)
        cHand, _ = hand_wrapper(geom_utils.matrix_to_se3(cTh), hA)

        iHand = mesh_utils.render_mesh(cHand, cameras)
        image1 = image_utils.blend_images(iHand['image'], mask=iHand['mask'], bg=data['image'])

        gif = mesh_utils.render_geom_rot(hHand, time_len=3, scale_geom=True)
        
        image_list[1].append(image1)
        image_list[2].append(gif[0])
        image_list[3].append(gif[1])

    with open(vis_dir + '/opt.pkl', 'wb') as fp:
        pickle.dump({'hA': hA_list, 'cTh': cTh_list}, fp)
    vis_hA(hA_list, vis_dir + '/opt', 'opt', range(6))

    for n, img_list in zip(name_list, image_list):
        image_utils.save_gif(img_list, osp.join(vis_dir, name + '_%s' % n))
    file_list = [osp.join(vis_dir, name + '_%s.gif' % n) for n in name_list]
    return file_list



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    zi = lfilter_zi(b, a)

    y = lfilter(b, a, data, zi=zi*data[0])[0]

    # y = lfilter_zi(b, a, data)
    return y


def low_pass_filter():
    win_size = 30
    order = 3
    # Creating the data for filteration
    dataloader = get_dataloader(seq)
    hA_list = []
    for data in dataloader:
        hA_list.append(data['hA'][0].cpu().detach().numpy())

    D = hA_list[0].shape[-1]
    T = len(hA_list)         # value taken in seconds
    t = np.linspace(0, T, T, endpoint=False)
    
    y_list = []
    for d in range(D):
        data = [hA_list[t][d] for t in range(len(hA_list))]
        # Filtering and plotting
        y = savgol_filter(data, win_size, order) # window size 51, polynomial order 3  (T, 
        y_list.append(y)
    y_list = np.array(y_list).transpose([1, 0])  
    print(y_list.shape)
    # y = butter_lowpass_filter(data, cutoff, fs, order)

    vis_hA(y_list, vis_dir + '/low_pass', 'low pass', range(6))

    vis_dataset(dataloader, 'low_pass', torch.FloatTensor(hA_list))


def low_pass_filter_via_mat():
    win_size = 10
    order = 3
    # Creating the data for filteration
    dataloader = get_dataloader(seq)
    hA_list = []
    for data in dataloader:
        hA_list.append(data['hA'][0].cpu().detach().numpy())

    D = hA_list[0].shape[-1]
    T = len(hA_list)         # value taken in seconds
    t = np.linspace(0, T, T, endpoint=False)
    
    y_list = []
    for d in range(0, D, 3):
        data = [hA_list[t][d:d+3] for t in range(len(hA_list))]  # T, 3?
        mat = geom_utils.axis_angle_t_to_matrix(torch.FloatTensor(data), homo=False)  # T, 3, 3
        print(mat.shape)
        mat = mat.detach().numpy().reshape([T, -1])  # T, 9

        # Filtering and plotting
        mat = [savgol_filter(mat[:, m], win_size, order) for m in range(9)] # 9, T
        mat = np.array(mat).transpose([1, 0]) 
        print(mat.shape)
        mat = mat.reshape([T, 3, 3])

        y = scipy.spatial.transform.Rotation.from_matrix(mat).as_rotvec()
        print(y.shape)
        y_list.append(y)  # (T, 3,)
    y_list = np.concatenate(y_list, axis=-1)  # (T, 3*45)
    print(y_list.shape)
    # y = butter_lowpass_filter(data, cutoff, fs, order)

    vis_hA(y_list, vis_dir + '/rot_low_pass', 'low pass', range(6))

    vis_dataset(dataloader, 'rot_low_pass', torch.FloatTensor(hA_list))

            
if __name__ == '__main__':
    # optimize(wt=10)
    # low_pass_filter()
    low_pass_filter_via_mat()
    vis_inp()