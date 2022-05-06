import os.path as osp
import glob
import torch
import pickle
import numpy as np

import os
import tqdm

from pytorch3d.structures import Meshes
from jutils import mesh_utils, geom_utils, image_utils

from utils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o

save_dir =vis_dir = '/glusterfs/yufeiy2/vhoi/vis/'
data_dir = '/glusterfs/yufeiy2/download_data/YCBAffordance'
device = 'cuda:0'

def Visualize():
    grasps = glob.glob(data_dir + '/grasps/obj_*')
    # grasps = glob.glob(data_dir + '/grasps_extending_symmetry/grasp_*')
    grasps.sort()
    np.random.seed(1)
    np.random.shuffle(grasps)

    hand_wrapper = ManopthWrapper().to(device)

    for i, grasp in tqdm.tqdm(enumerate(grasps)):
        with open(grasp, 'rb') as f:
            hand = pickle.load(f, encoding='latin')

        filename = data_dir + '/models/' + hand['body'][36:]

        objname = str.split(filename, 'nontextured_transformed.wrl')[0] + 'textured.obj'
        obj = fast_load_obj(open(objname, 'rb'))[0]
        # obj = load_mesh(objname)

        obj_verts = obj['vertices']
        obj_faces = obj['faces']

        hA = hand_wrapper.pca_to_pose(torch.FloatTensor(hand['pca_poses']).to(device), add_mean=False)
        print(hand['mano_trans'].shape, hand['pca_manorot'].shape)
        rot, trans = torch.FloatTensor([hand['pca_manorot']]).reshape(1, 3), torch.FloatTensor(hand['mano_trans']).reshape(1, 3)
        rot, trans = cvt_axisang_t_i2o(rot, trans)
        print(rot.shape, trans.shape)

        rt = geom_utils.axis_angle_t_to_matrix(rot, trans).to(device)
        print(rt.shape, hA.shape)
        hand, _ = hand_wrapper(rt, hA)
        # hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(posesnew), th_trans=torch.FloatTensor(mano_trans))
        # hand_vertices = hand_vertices.cpu().data.numpy()[0]/1000
        # hand_faces = mano_layer_right.th_faces.cpu().data.numpy()

        # plot_hand_w_object(obj_verts, obj_faces, hand_vertices, hand_faces)
        # print(obj_verts.shape, obj_faces.shape, hand_vertices.shape, hand_faces.shape)
        obj = Meshes([torch.FloatTensor(obj_verts)], [torch.LongTensor(obj_faces)]).to(device)
        # hand = Meshes([torch.FloatTensor(hand_vertices)], [torch.LongTensor(hand_faces)])
        hoi = mesh_utils.join_scene([obj, hand]).to(device)
        image_utils.save_gif(mesh_utils.render_geom_rot(hoi, scale_geom=True), osp.join(save_dir, '%d' % i))
        if i > 0:
            break

def load_yucba(grasp, hand_wrapper, data_dir, device='cuda:0'):
    with open(osp.join(data_dir, 'grasps/%s.pickle' % grasp), 'rb') as f:
        hand = pickle.load(f, encoding='latin')

    filename = data_dir + '/models/' + hand['body'][36:]

    objname = str.split(filename, 'nontextured_transformed.wrl')[0] + 'model_watertight_2000def.obj'
    obj = fast_load_obj(open(objname, 'rb'))[0]

    obj_verts = obj['vertices']
    obj_faces = obj['faces']

    hA = hand_wrapper.pca_to_pose(torch.FloatTensor(hand['pca_poses']).to(device), add_mean=False)

    rot, trans = torch.FloatTensor([hand['pca_manorot']]).reshape(1, 3), torch.FloatTensor(hand['mano_trans']).reshape(1, 3)
    rot, trans = cvt_axisang_t_i2o(rot, trans)

    oTh = geom_utils.axis_angle_t_to_matrix(rot, trans).to(device)
    obj = Meshes([torch.FloatTensor(obj_verts)], [torch.LongTensor(obj_faces)]).to(device)
    hObj = mesh_utils.apply_transform(obj, geom_utils.inverse_rt(mat=oTh, return_mat=True))
    return  hObj, hA


def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    and taken from ObMan dataset (https://github.com/hassony2/obman) 
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.
    vertices with the same position but different normals or uvs
    are split into multiple vertices.
    colors are discarded.
    parameters
    ----------
    file_obj : file object
                   containing a wavefront file
    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []
    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes


def make_list():
    grasp_file = data_dir + '/Sets/manual_grasp.txt'
    index_list = [e.strip().split('.pickle')[0] for e in open(grasp_file)]

    with open(osp.join(data_dir, 'all.txt'), 'w') as fp:
        for index in index_list:
            fp.write('%s\n' % index)

    np.random.seed(123)
    np.random.shuffle(index_list)
    num = len(index_list) // 10
    with open(osp.join(data_dir, 'test_full.txt'), 'w') as fp:
        for index in index_list[:num]:
            fp.write('%s\n' % index)
    with open(osp.join(data_dir, 'train_full.txt'), 'w') as fp:
        for index in index_list[num:]:
            fp.write('%s\n' % index)


if __name__ == '__main__':
    # Visualize()
    make_list()