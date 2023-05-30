# https://github.com/shreyashampali/ho3d/blob/master/eval.py
from __future__ import print_function, unicode_literals
import os
import os.path as osp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
import pip
import argparse
import json

from scipy.linalg import orthogonal_procrustes


class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds
    
def verts2pcd(verts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = gt.compute_point_cloud_distance(pr) # closest dist for each gt point
    d2 = pr.compute_point_cloud_distance(gt) # closest dist for each pred point
    # d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr) # closest dist for each gt point
    # d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt) # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall

def align_sc_tr(mtx1, mtx2):
    """ Align the 3D joint location with the ground truth by scaling and translation """

    predCurr = mtx2.copy()
    # normalize the predictions
    s = np.sqrt(np.sum(np.square(predCurr[4] - predCurr[0])))
    if s>0:
        predCurr = predCurr / s

    # get the scale of the ground truth
    sGT = np.sqrt(np.sum(np.square(mtx1[4] - mtx1[0])))

    # make predictions scale same as ground truth scale
    predCurr = predCurr * sGT

    # make preditions translation of the wrist joint same as ground truth
    predCurrRel = predCurr - predCurr[0:1, :]
    preds_sc_tr_al = predCurrRel + mtx1[0:1, :]

    return preds_sc_tr_al




def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:
    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text


def createHTML(outputDir, curve_list):
    curve_data_list = list()
    for item in curve_list:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(item.x_data, item.y_data)
        ax.set_xlabel(item.x_label)
        ax.set_ylabel(item.y_label)
        img_path = os.path.join(outputDir, "img_path_path.png")
        plt.savefig(img_path, bbox_inches=0, dpi=300)

        # write image and create html embedding
        data_uri1 = open(img_path, 'rb').read().encode('base64').replace('\n', '')
        img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri1)
        curve_data_list.append((item.text, img_tag1))

        os.remove(img_path)

    htmlString = '''<!DOCTYPE html>
    <html>
    <body>
    <h1>Detailed results:</h1>'''

    for i, (text, img_embed) in enumerate(curve_data_list):
        htmlString += '''
        <h2>%s</h2>
        <p>
        <img border="0" %s alt="FROC" width="576pt" height="432pt">
        </p>
        <p>Raw curve data:</p>
        
        <p>x_axis: <small>%s</small></p>
        <p>y_axis: <small>%s</small></p>
        
        ''' % (text, img_embed, curve_list[i].x_data, curve_list[i].y_data)

    htmlString += '''
    </body>
    </html>'''

    htmlfile = open(os.path.join(outputDir, "scores.html"), "w")
    htmlfile.write(htmlString)
    htmlfile.close()


def _search_pred_file(pred_path, pred_file_name):
    """ Tries to select the prediction file. Useful, in case people deviate from the canonical prediction file name. """
    pred_file = os.path.join(pred_path, pred_file_name)
    if os.path.exists(pred_file):
        # if the given prediction file exists we are happy
        return pred_file

    print('Predition file "%s" was NOT found' % pred_file_name)

    # search for a file to use
    print('Trying to locate the prediction file automatically ...')
    files = [os.path.join(pred_path, x) for x in os.listdir(pred_path) if x.endswith('.json')]
    if len(files) == 1:
        pred_file_name = files[0]
        print('Found file "%s"' % pred_file_name)
        return pred_file_name
    else:
        print('Found %d candidate files for evaluation' % len(files))
        raise Exception('Giving up, because its not clear which file to evaluate.')


def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def main(gt_path, pred_path, output_dir=None, pred_file_name=None, set_name=None):
    if output_dir is None:
        output_dir = osp.dirname(pred_path)
    # load eval annotations
    xyz_list, verts_list = json_load(gt_path)

    # load predicted values
    with open(pred_path, 'r') as fi:
        pred = json.load(fi)

    assert len(pred) == 2, 'Expected format mismatch.'
    # assert len(pred[0]) - len(xyz_list) >= -2, f'Expected format mismatch. {len(pred[0])} {len(xyz_list)}'
    # assert len(pred[1]) - len(xyz_list) >= -2, f'Expected format mismatch.{len(pred[1])} {len(xyz_list)}'

    # init eval utils
    eval_xyz, eval_xyz_procrustes_aligned, eval_xyz_sc_tr_aligned = EvalUtil(), EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]

    shape_is_mano = None
    rng = range(len(pred[0]))

    # iterate over the dataset once
    for idx in rng:

        xyz, verts = xyz_list[idx], verts_list[idx]
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        xyz_pred, verts_pred = pred[0][idx], pred[1][idx]
        xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

        # Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )

        if shape_is_mano is None:
            if verts_pred.shape[0] == verts.shape[0]:
                shape_is_mano = True
            else:
                shape_is_mano = False

        if shape_is_mano:
            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred
            )

        # scale and translation aligned predictions for xyz
        xyz_pred_sc_tr_aligned = align_sc_tr(xyz, xyz_pred)
        eval_xyz_sc_tr_aligned.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred_sc_tr_aligned
        )

        # align predictions
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        if shape_is_mano:
            verts_pred_aligned = align_w_scale(verts, verts_pred)
        else:
            # use trafo estimated from keypoints
            trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)
            verts_pred_aligned = align_by_trafo(verts_pred, trafo)

        # Aligned errors
        eval_xyz_procrustes_aligned.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred_aligned
        )

        if shape_is_mano:
            eval_mesh_err_aligned.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred_aligned
            )

        # F-scores
        l, la = list(), list()
        for t in f_threshs:
            # for each threshold calculate the f score and the f score of the aligned vertices
            f, _, _ = calculate_fscore(verts, verts_pred, t)
            l.append(f)
            f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
            la.append(f)
        f_score.append(l)
        f_score_aligned.append(la)

    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_procrustes_al_mean3d, _, xyz_procrustes_al_auc3d, pck_xyz_procrustes_al, thresh_xyz_procrustes_al = eval_xyz_procrustes_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP PROCRUSTES ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_procrustes_al_auc3d, xyz_procrustes_al_mean3d * 100.0))

    xyz_sc_tr_al_mean3d, _, xyz_sc_tr_al_auc3d, pck_xyz_sc_tr_al, thresh_xyz_sc_tr_al = eval_xyz_sc_tr_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP SCALE-TRANSLATION ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_sc_tr_al_auc3d, xyz_sc_tr_al_mean3d * 100.0))


    if shape_is_mano:
        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_vert3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH ALIGNED results:')
        print('auc=%.3f, mean_vert3d_avg=%.2f cm\n' % (mesh_al_auc3d, mesh_al_mean3d * 100.0))
    else:
        mesh_mean3d, mesh_auc3d, mesh_al_mean3d, mesh_al_auc3d = -1.0, -1.0, -1.0, -1.0

        pck_mesh, thresh_mesh = np.array([-1.0, -1.0]), np.array([0.0, 1.0])
        pck_mesh_al, thresh_mesh_al = np.array([-1.0, -1.0]), np.array([0.0, 1.0])

    print('F-scores')
    f_out = list()
    f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
    for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
        print('F@%.1fmm = %.3f' % (t*1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t*1000, fa.mean()))
        f_out.append('f_score_%d: %f' % (round(t*1000), f.mean()))
        f_out.append('f_al_score_%d: %f' % (round(t*1000), fa.mean()))

    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        xyz_mean3d *= 100
        xyz_procrustes_al_mean3d *= 100
        xyz_sc_tr_al_mean3d *= 100
        fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
        fo.write('xyz_auc3d: %f\n' % xyz_auc3d)
        fo.write('xyz_procrustes_al_mean3d: %f\n' % xyz_procrustes_al_mean3d)
        fo.write('xyz_procrustes_al_auc3d: %f\n' % xyz_procrustes_al_auc3d)
        fo.write('xyz_scale_trans_al_mean3d: %f\n' % xyz_sc_tr_al_mean3d)
        fo.write('xyz_scale_trans_al_auc3d: %f\n' % xyz_sc_tr_al_auc3d)

        mesh_mean3d *= 100
        mesh_al_mean3d *= 100
        fo.write('mesh_mean3d: %f\n' % mesh_mean3d)
        fo.write('mesh_auc3d: %f\n' % mesh_auc3d)
        fo.write('mesh_al_mean3d: %f\n' % mesh_al_mean3d)
        fo.write('mesh_al_auc3d: %f\n' % mesh_al_auc3d)
        for t in f_out:
            fo.write('%s\n' % t)
    print('Scores written to: %s' % score_path)

    # createHTML(
    #     output_dir,
    #     [
    #         curve(thresh_xyz*100, pck_xyz, 'Distance in cm', 'Percentage of correct keypoints', 'PCK curve for aligned keypoint error'),
    #         curve(thresh_xyz_procrustes_al*100, pck_xyz_procrustes_al, 'Distance in cm', 'Percentage of correct keypoints', 'PCK curve for procrustes aligned keypoint error'),
    #         curve(thresh_xyz_sc_tr_al * 100, pck_xyz_sc_tr_al, 'Distance in cm',
    #               'Percentage of correct keypoints', 'PCK curve for scale-translation aligned keypoint error'),
    #         curve(thresh_mesh*100, pck_mesh, 'Distance in cm', 'Percentage of correct vertices', 'PCV curve for mesh error'),
    #         curve(thresh_mesh_al*100, pck_mesh_al, 'Distance in cm', 'Percentage of correct vertices', 'PCV curve for aligned mesh error')
    #     ]
    # )

    print('Evaluation complete.')
    metric_list = [xyz_mean3d, xyz_auc3d, xyz_procrustes_al_mean3d, xyz_procrustes_al_auc3d, xyz_sc_tr_al_mean3d, xyz_sc_tr_al_auc3d, mesh_mean3d, mesh_auc3d, mesh_al_mean3d, mesh_al_auc3d]
    return metric_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--input_dir', type=str,
                        help='Path to where prediction the submited result and the ground truth is.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to where the eval result should be.')
    parser.add_argument('--gt', type=str,
                        help='Path to where the eval result should be.')
    parser.add_argument('--pred', type=str,
                        help='Path to where the eval result should be.')
    parser.add_argument('--pred_file_name', type=str, default='pred.json',
                        help='Name of the eval file.')
    parser.add_argument('--version', type=str, choices=['v2', 'v3'],
                        help='HO3D version', default='v2')
    args = parser.parse_args()

    # call eval
    main(
        gt_path=args.gt,
        pred_path=args.pred,
        output_dir=args.output_dir,
    )