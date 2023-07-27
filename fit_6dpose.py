import os
import os.path as osp
import torch
import torch.nn as nn
import wandb
from torch.utils.data.dataloader import DataLoader
from dataio import get_data

from jutils import slurm_utils, mesh_utils, plot_utils, image_utils, geom_utils


device = 'cuda:0'
def find_optimal_poses(template, dataloader:DataLoader, num_iterations=2, num_initializations=2000, batch_size=500):
    """
    Find optimal poses for the object by shooting method
    """
    dataset = dataloader.dataset
    oTh = torch.zeros([len(dataset), 4, 4], device=device)
    for t in range(len(dataset)):
        oTh_t = optimize_frame(dataset[t])
        oTh[t] = oTh_t

    def optimize_frame(frame):
        oTh_all_rot = geom_utils.random_rotations(num_initializations, device=device)
        oTh_all_tsl = torch.zeros([num_initializations, 3], device=device)

        loss_list = []

        # split into batch_size chunk and SGD each batch
        for start in range(0, num_initializations, batch_size):
            end = min(start + batch_size, num_initializations)
            oTh_all_rot_batch = oTh_all_rot[start:end]
            oTh_all_tsl_batch = oTh_all_tsl[start:end]

            # optimize
            oTh_all_rot_batch, oTh_all_tsl_batch, loss_batch = optimize_poses(
                template, frame, oTh_all_rot_batch, oTh_all_tsl_batch, num_iterations=num_iterations
            )
            oTh_all_rot[start:end] = oTh_all_rot_batch
            oTh_all_tsl[start:end] = oTh_all_tsl_batch
            loss_list.append(loss_batch)
        
        # find the best pose
        loss_list = torch.cat(loss_list, dim=0)
        best_idx = torch.argmin(loss_list)
        oTh_best_rot = oTh_all_rot[best_idx]
        oTh_best_tsl = oTh_all_tsl[best_idx]
        
        oTh = geom_utils.rt_to_homo(oTh_best_rot, oTh_best_tsl, )

        return oTh

    return oTh


def optimize_poses(template, frame, oTh_rot, oTh_tsl, num_iterations):
    idx, sample, gt = frame

    oTh_param = nn.Parameter()
    model = PoseOptimizer(
        ref_image=mask,
        vertices=vertices,
        faces=faces,
        textures=textures,
        rotation_init=matrix_to_rot6d(rotations_init),
        translation_init=translations_init,
        num_initializations=num_initializations,
        K=camintr_roi,
    )



def main_function(args):
    # get 2D evidences, and 3D CAD models
    dataset, val_dataset = get_data(args, return_val=True, val_downscale=args.data.get('val_downscale', 4.0))
    dataloader = DataLoader(dataset, batch_size=args.bs) #, collate_fn=mesh_utils.collate_meshes)
    template = load_mesh()

    # initialize pose
    # Increase these parameters if fit looks bad.
    num_iterations = 2
    num_initializations = 2000

    # Reduce batch size if your GPU runs out of memory.
    batch_size = 500

    object_parameters = find_optimal_poses(
        template, dataloader,
        num_iterations=num_iterations, num_initializations=num_initializations,
        batch_size=batch_size,
    )

    # create_model? 
    

    coarse_loss_weights = {
            "lw_inter": 1,
            "lw_depth": 0,
            "lw_sil_obj": 1.0,
            "lw_sil_hand": 0.0,
            "lw_collision": 0.0,
            "lw_contact": 0.0,
            "lw_scale_hand": 0.001,
            "lw_scale_obj": 0.001,
            "lw_v2d_hand": 50,
            "lw_smooth_hand": 2000,
            "lw_smooth_obj": 2000,
            "lw_pca": 0.004,
        }

    # joint optimization
    # Coarse hand-object fitting
    model, loss_evolution, imgs = optimize_hand_object(
    object_parameters=object_parameters,
    hand_proj_mode="persp",
    objvertices=obj_verts_can,
    objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
    optimize_mano=True,
    optimize_object_scale=True,
    loss_weights=coarse_loss_weights,
    image_size=image_size,
    num_iterations=coarse_num_iterations + 1,  # Increase to get more accurate initializations
    images=np.stack(images_np),
    camintr=camintr_nc,
    state_dict=None,
    viz_step=coarse_viz_step,
    viz_folder=step2_viz_folder,
)
    # refine opitization? 

    finegrained_loss_weights = {
        "lw_inter": 1,
        "lw_depth": 0,
        "lw_sil_obj": 1.0,
        "lw_sil_hand": 0.0,
        "lw_collision": 0.001,
        "lw_contact": 1.0,
        "lw_scale_hand": 0.001,
        "lw_scale_obj": 0.001,
        "lw_v2d_hand": 50,
        "lw_smooth_hand": 2000,
        "lw_smooth_obj": 2000,
        "lw_pca": 0.004,
    }

    # Refine hand-object fitting
    step3_folder = os.path.join(sample_folder, "jointoptim_step3")
    step3_viz_folder = os.path.join(step3_folder, "viz")
    model_fine, loss_evolution, imgs = optimize_hand_object(
        person_parameters=person_parameters,
        object_parameters=object_parameters,
        hand_proj_mode="persp",
        objvertices=obj_verts_can,
        objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
        optimize_mano=True,
        optimize_object_scale=True,
        loss_weights=finegrained_loss_weights,
        image_size=image_size,
        num_iterations=finegrained_num_iterations + 1,
        images=np.stack(images_np),
        camintr=camintr_nc,
        state_dict=model.state_dict(),
        viz_step=finegrained_viz_step, 
        viz_folder=step3_viz_folder,
    )

    return 