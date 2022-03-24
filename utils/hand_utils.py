# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import pickle
import os.path as osp
from typing import Tuple
import numpy as np
import smplx
import torch
import torch.nn as nn
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, Rotate, Translate
from pytorch3d.io import load_obj

from manopth.manolayer import ManoLayer
from manopth.tensutils import th_with_zeros, th_posemap_axisang
from jutils import geom_utils


def get_nTh(r=0.2, center=None, hA=None, hand_wrapper=None, inverse=False):
    """
    
    Args:
        center: (N, 3?)
        hA (N, 45 ): Description
        r (float, optional): Description
    Returns: 
        (N, 4, 4)
    """
    # add a dummy batch dim
    if center is None:
        hJoints = hand_wrapper(None, hA, mode='inner')[1]
        start = 5
        center = hJoints[:, start]  #
    device = center.device
    N = len(center)
    y = 0.08
    center = center + torch.FloatTensor([0, -y, 0]).unsqueeze(0).to(device)

    # (x - center) / r
    mat = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
    mat[..., :3, :3] /= r
    mat[..., :3, 3] = -center / r

    if inverse:
        mat = geom_utils.inverse_rt(mat=mat, return_mat=True)
    return mat


def load_pkl(pkl_file, res_list=None):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


def get_mean_pose(model='mano', tensor=True, device='cpu'):
    """:return: (1, 45) """
    if model == 'mano':
        mean_pose = np.array(
            [[0.11167871, -0.04289218, 0.41644183, 0.10881133, 0.06598568, 0.75622,
              -0.09639297, 0.09091566, 0.18845929, -0.11809504, -0.05094385, 0.5295845,
              -0.14369841, -0.0552417, 0.7048571, -0.01918292, 0.09233685, 0.3379135,
              -0.45703298, 0.19628395, 0.6254575, -0.21465237, 0.06599829, 0.50689423,
              -0.36972436, 0.06034463, 0.07949023, -0.1418697, 0.08585263, 0.63552827,
              -0.3033416, 0.05788098, 0.6313892, -0.17612089, 0.13209307, 0.37335458,
              0.8509643, -0.27692273, 0.09154807, -0.49983943, -0.02655647, -0.05288088,
              0.5355592, -0.04596104, 0.27735803]], dtype=np.float32
        )
    else:
        raise NotImplementedError
    if tensor:
        mean_pose = torch.tensor(mean_pose, device=device, dtype=torch.float32)
    return mean_pose


class ManopthWrapper(nn.Module):
    def __init__(self, mano_path='/checkpoint/yufeiy2/pretrain_model/smplx/mano/', **kwargs):
        super().__init__()
        self.mano_layer_right = ManoLayer(
            mano_root=mano_path, side='right', use_pca=kwargs.get('use_pca', False), ncomps=kwargs.get('ncomps', 45),
            flat_hand_mean=kwargs.get('flat_hand_mean', True))
        self.metric = kwargs.get('metric', 1)
        
        self.register_buffer('hand_faces', self.mano_layer_right.th_faces.unsqueeze(0))
        self.register_buffer('hand_mean', torch.FloatTensor(self.mano_layer_right.smpl_data['hands_mean']).unsqueeze(0))
        self.register_buffer('t_mano', torch.tensor([[0.09566994, 0.00638343, 0.0061863]], dtype=torch.float32, ))
        self.register_buffer('th_selected_comps', torch.FloatTensor(self.mano_layer_right.smpl_data['hands_components']))
        self.register_buffer('inv_scale', 1. / torch.sum(self.th_selected_comps ** 2, dim=-1))  # (D, ))

        with open(osp.join(mano_path, 'contact_zones.pkl'), 'rb') as fp:
            contact = pickle.load(fp)['contact_zones']
            contact_list = []
            for ind, verts_idx in contact.items():
                contact_list.extend(verts_idx)
                self.register_buffer('contact_index_%d' % ind, torch.LongTensor(verts_idx))
        self.register_buffer('contact_index' , torch.LongTensor(contact_list))

        # load uv map
        rtn = load_obj(osp.join(mano_path, 'open_hand_uv.obj'))
        self.register_buffer('verts_uv', rtn[2].verts_uvs[None])
        self.register_buffer('faces_uv', rtn[1].textures_idx[None])
    
    def to_palm(self, rot, hA, add_pca=False, return_mesh=True):
        if add_pca:
            hA = hA + self.hand_mean
        
        art_pose = torch.cat([rot, hA], -1)
        verts, joints, faces = self._forward_layer(art_pose, torch.zeros_like(rot))

        textures = torch.ones_like(verts)

        verts = verts - joints[:, 5:6]  # center at palm
        # palmTh = geom_utils.axis_angle_t_to_matrix(rot, -joints[:, 5])
        palmTh = geom_utils.axis_angle_t_to_matrix(*cvt_axisang_t_i2o(rot, -joints[:, 5]))
        joints = joints - joints[:, 5:6]

        if return_mesh:
            return Meshes(verts, faces, TexturesVertex(textures)), joints, palmTh
        else:
            return verts, faces, textures, joints, palmTh


    def forward(self, glb_se3, art_pose, axisang=None, trans=None, return_mesh=True, 
        mode='outer', texture='uv', **kwargs) -> Tuple[Meshes, torch.Tensor]:
        # return 
    # def __call__(self, glb_se3, art_pose, axisang=None, trans=None, return_mesh=True, mode='outer', **kwargs):
        N = len(art_pose)
        device = art_pose.device

        if mode == 'outer':
            if axisang is None:
                axisang = torch.zeros([N, 3], device=device)
            if trans is None:
                trans = torch.zeros([N, 3], device=device)
            if art_pose.size(-1) == 45:
                art_pose = torch.cat([axisang, art_pose], -1)
            verts, joints, faces = self._forward_layer(art_pose, trans)

            if glb_se3 is not None:
                mat_rt = geom_utils.se3_to_matrix(glb_se3)
                trans = Transform3d(matrix=mat_rt.transpose(1, 2))
                verts = trans.transform_points(verts)
                joints = trans.transform_points(joints)
        else:  # inner translation
            if axisang is None:
                axisang = torch.zeros([N, 3], device=device)
            art_pose = torch.cat([axisang, art_pose], -1)
            # if axisang is not None:
                # art_pose = torch.cat([axisang, art_pose], -1)
            if trans is None:
                trans = torch.zeros([N, 3], device=device)
            # if art_pose.size(-1) == 45:
            verts, joints, faces = self._forward_layer(art_pose, trans, **kwargs)

        if texture == 'verts':
            textures = torch.ones_like(verts)
            textures = TexturesVertex(textures)
        elif torch.is_tensor(texture):
            textures = TexturesUV(texture, self.faces_uv.repeat(N, 1, 1), self.verts_uv.repeat(N, 1, 1))

        else:
            raise NotImplementedError
        if return_mesh:
            return Meshes(verts, faces, textures), joints
        else:
            return verts, faces, textures, joints

    def pose_to_pca(self, pose, ncomps=45):
        """
        :param pose: (N, 45)
        :return: articulated pose: (N, pca)
        """
        pose = pose - self.hand_mean
        components = self.th_selected_comps[:ncomps]  # (D, 45)
        scale = self.inv_scale[:ncomps]

        coeff = pose.mm(components.transpose(0, 1)) * scale.unsqueeze(0)
        return coeff

    def pca_to_pose(self, pca):
        """
        :param pca: (N, Dpca)
        :return: articulated pose: (N, 45)
        """
        # Remove global rot coeffs
        ncomps = pca.size(-1)
        theta = pca.mm(self.th_selected_comps[:ncomps]) + self.hand_mean
        return theta

    def cTh_transform(self, hJoints: torch.Tensor, cTh: torch.Tensor) -> Transform3d:
        """
        :param hMeshes: meshes in hand space
        :param hJoints: joints, in shape of (N, J, 3)
        :param cTh: (N, 6) intrinsic and extrinsic for a weak perspaective camera (s, x, y, rotaxisang)
        :return: cMeshes: meshes in full perspective camera space.
            se3 = geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot, ))
            mesh, j3d = hand_wrapper(se3, art_pose)
            f = 200.
            camera = PerspectiveCameras(focal_length=f,  device=device, T=torch.FloatTensor([[0, 0, .1]]).cuda())
            translate = torch.stack([tx, ty, f/s], -1)  # N, 1, 3
            mesh = mesh.update_padded(mesh.verts_padded() - j3d[:, start:start + 1] + translate)
        """
        device = hJoints.device
        if cTh.size(-1) == 7:
            f, s, tx, ty, axisang = torch.split(cTh, [1, 1, 1, 1, 3], dim=-1)
            translate = self.metric * torch.cat([tx, ty, f / s], -1)  # N, 3
            rot = Rotate(geom_utils.axis_angle_t_to_matrix(axisang, homo=False).transpose(1, 2), device=device)

            start = 5
            hJoints = rot.transform_points(hJoints)
            offset = Translate(-hJoints[:, start] + translate, device=device)

            cTh_transform = rot.compose(offset)  # R X + t
        else:
            cTh_transform = geom_utils.rt_to_transform(cTh)
        return cTh_transform

    def _forward_layer(self, pose, trans, **kwargs):
        verts, joints = self.mano_layer_right(pose, th_trans=trans, **kwargs) # in MM
        verts /= (1000 / self.metric)
        joints /= (1000 / self.metric)

        faces = self.hand_faces.repeat(verts.size(0), 1, 1)

        return verts, joints, faces

    def pose_to_transform(self, hA, include_wrist=True):
        """
        :param hA: (N, (3+)45)
        :param include_wrist:
        :return: (N, (3+)J, 4, 4)
        """
        N = hA.size(0)
        device = hA.device

        if not include_wrist:
            zeros = torch.zeros([N, 3], device=device)
            hA = torch.cat([zeros, hA], -1)

        th_pose_map, th_rot_map = th_posemap_axisang(hA)
        root_rot = th_rot_map[:, :9].view(N, 3, 3)
        th_rot_map = th_rot_map[:, 9:]

        # Full axis angle representation with root joint
        th_shapedirs = self.mano_layer_right.th_shapedirs
        th_betas = self.mano_layer_right.th_betas
        th_J_regressor = self.mano_layer_right.th_J_regressor
        th_v_template = self.mano_layer_right.th_v_template

        th_v_shaped = torch.matmul(th_shapedirs,
                                   th_betas.transpose(1, 0)).permute(
                                       2, 0, 1) + th_v_template
        th_j = torch.matmul(th_J_regressor, th_v_shaped).repeat(
            N, 1, 1)

        # Global rigid transformation
        root_j = th_j[:, 0, :].contiguous().view(N, 3, 1)  # wrist coord
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))  # homogeneousr [R, t]

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]  # (1, 16, 4, 4)
        th_results_global = th_results

        th_jtr = th_results_global
        # todo
        th_jtr = th_jtr[:, [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]]
        return th_jtr


class ManoWrapper:
    def __init__(self, cfg):
        self.mano = smplx.create(cfg.HAND.MANO_PATH, 'mano')
        # self.mano = smplx.build_layer(cfg.HAND.MANO_PATH, 'mano')
        self.hand_info = load_pkl(cfg.HAND_INFO_FILE)

        self.right_hand_faces_local = self.hand_info['right_hand_faces_local']

    def __call__(self, se3, hand_type='right_hand', shape_param=None, pose_param=None, return_mesh=True, **kwargs):
        """
        # joints index order - panoptic , rot form: axis angle
        :param return_mesh: global?
        :param kwargs:
        :return: joints in panoptic order. verts /faces in MANO_RIGHT?? order
        """
        N = se3.size(0)
        device = se3.device

        axisangle, transl = geom_utils.se3_to_axis_angle_t(se3)
        hand_out = self.mano(betas=shape_param, hand_pose=pose_param, global_orient=axisangle, transl=transl,**kwargs)
        verts = hand_out['vertices']

        smplx_hand_to_panoptic = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        hand_joints = hand_out['joints'][:, smplx_hand_to_panoptic, :]

        faces = self.hand_info['right_hand_faces_local']
        faces = faces.expand(N, faces.size(1), 3).to(device)

        if hand_type == 'left_hand':  # flip left hand
            verts[:, 0] *= -1
            faces = faces[:, ::-1]
            hand_joints[:, 0] *= -1

        textures = torch.ones_like(verts)
        if return_mesh:
            return  Meshes(verts, faces, TexturesVertex(textures))
        else:
            return verts, faces, textures


def cvt_axisang_t_i2o(axisang, trans):
    """+correction: t_r - R_rt_r. inner to outer"""
    trans += get_offset(axisang)

    return axisang, trans


def cvt_axisang_t_o2i(axisang, trans):
    """-correction: t_r, R_rt_r. outer to inner"""
    trans -= get_offset(axisang)
    return axisang, trans


def get_offset(axisang):
    """
    :param axisang: (N, 3)
    :return: trans: (N, 3) = r_r - R_r t_r
    """
    device = axisang.device
    N = axisang.size(0)
    t_mano = torch.tensor([[0.09566994, 0.00638343, 0.0061863]], dtype=torch.float32, device=device).repeat(N, 1)
    # t_r = torch.tensor([[0.09988064, 0.01178287,  -0.01959994]], dtype=torch.float32, device=device).repeat(N, 1)
    rot_r = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)  # N, 3, 3
    delta = t_mano - torch.matmul(rot_r, t_mano.unsqueeze(-1)).squeeze(-1)
    return delta


def cvt_mano_to_smplx(axisang, trans):
    """
    :param axisang: R_r
    :param trans: t_g
    :return:
    """
    device = axisang.device
    N = axisang.size(0)
    t_r = torch.tensor([[0.0957, 0.0064, 0.0062]], dtype=torch.float32, device=device).repeat(N, 1)  # N ,3
    # rot_r = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)  # N, 3, 3
    # delta = t_r - torch.matmul(rot_r, t_r.unsqueeze(-1)).squeeze(-1)
    delta = get_offset(axisang)

    trans = trans + delta

    return axisang, trans


def transform_points(trans, points):
    """
    :param trans: (N, 4, 4)
    :param points: (N, P, 3)
    :return: (N, 3)
    """
    device = trans.device
    trans = Transform3d(matrix=trans.transpose(1, 2), device=device)
    points = trans.transform_points(points)
    return points


def fk_marker(hTjs, jOffset):
    N, J, _, _ = hTjs.size()
    hOffset = transform_points(hTjs.view(N * J, 4, 4), jOffset.view(N * J, 1, 3))
    hOffset = hOffset.view(N, J, 3)
    hOffset = hOffset.view(N, J, 3)
    red = torch.zeros_like(hOffset)
    red[..., 0] = 1

    marker = mesh_utils.pc_to_cubic_meshes(hOffset, red)
    return marker, hOffset

if __name__ == '__main__':
    from nnutils import mesh_utils, image_utils
    wrapper = ManopthWrapper()
    N = 1
    device = 'cuda'
    save_dir = '/checkpoint/yufeiy2/hoi_output/mano_pca'

    # hA = torch.ones([N, 45], device=device) + wrapper.hand_mean
    # pca = wrapper.pose_to_pca(hA)
    pca = torch.ones([N, 45], device=device)
    hA = wrapper.pca_to_pose(pca)
    hA_hat = wrapper.pca_to_pose(wrapper.pose_to_pca(hA))

    basis = wrapper.th_selected_comps  # D, 45
    print('basis', torch.matmul(basis, basis.transpose(0, 1)))


    zeros = torch.zeros([N, 3], device=device)

    mesh = wrapper(None,  hA, zeros, mode='inner')
    image_list = mesh_utils.render_geom_rot(mesh, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, 'hA'))

    mesh, _ = wrapper(None,  hA_hat, zeros, mode='inner')
    image_list = mesh_utils.render_geom_rot(mesh, scale_geom=True)
    image_utils.save_gif(image_list, osp.join(save_dir, 'hA_hat'))
    # pca = -torch.ones([N, 45], device=device)
    # hA = wrapper.pca_to_pose(pca)
