

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=ho3d_yana_det_why/soft_cond_\${data.index}_w\${training.w_diffuse}_\${data.len}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth} \
    training=diffuse novel_view=geom novel_view.diff_name="hand_ddpm_geom/train_seg_CondGeomGlide" \
    novel_view.loss.w_mask=0,1 novel_view.loss.w_depth=0,1 novel_view.loss.w_normal=0,1 \
    data=ho3d_det data.index=SM1_0360 training.w_diffuse=1e-2 data.len=2\
    environment.slurm=True environment.resume=False \




CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d/\${data.index}_w_\${training.w_diffuse}_clip\${training.clip}_geom_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth} \
    training=diffuse novel_view=geom novel_view.diff_name="geom/pretrained_ho3d_cam_train_seg_1_0.0001" \
    training.w_diffuse=0 training.render_full_frame=False \
    data=hoi4d \
    environment.slurm=True environment.resume=False  \


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d/\${data.index}_w_\${training.w_diffuse}_clip\${training.clip}_geom_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth} \
    training=diffuse novel_view=geom novel_view.diff_name="geom/pretrained_ho3d_cam_train_seg_1_0.0001" \
    novel_view.loss.w_mask=1 novel_view.loss.w_depth=1 novel_view.loss.w_normal=1 \
    training.w_diffuse=0.01,1e-3 \
    data=hoi4d data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159, \
    environment.slurm=True environment.resume=False \



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=train_seg_geom/\${data.index}_w_\${training.w_diffuse}_clip\${training.clip}_geom_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth} \
    training=diffuse novel_view=geom novel_view.diff_name="geom/pretrained_ho3d_cam_train_seg_1_0.0001" \
    novel_view.loss.w_mask=1 novel_view.loss.w_depth=1 novel_view.loss.w_normal=1 \
    training.w_diffuse=0.01,1e-3 \
    data=ho3d  data.len=2 data.index=SM2_0001_dt10,SMu1_0650_dt10,MDF10_1000_dt10 \
    environment.slurm=True environment.resume=False \




CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=train_seg_geom/w_\${training.w_diffuse}_clip\${training.clip}_geom_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth} \
    training=diffuse novel_view=geom novel_view.diff_name="geom/pretrained_ho3d_cam_train_seg_1_0.0001" \
    novel_view.loss.w_mask=1 novel_view.loss.w_depth=0,1 novel_view.loss.w_normal=0,1 \
    training.w_diffuse=0.01,1e-3 \
    data=ho3d  data.len=2 \
    environment.slurm=True environment.resume=False \



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=SM2only/clip\${training.clip}_geom_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth} \
    training=diffuse novel_view=geom novel_view.diff_name="geom/SM2_1_0.0001" \
    novel_view.loss.w_mask=1 novel_view.loss.w_depth=1 novel_view.loss.w_normal=1 \
    training.clip=10,100 \
    data=ho3d  data.len=2 \
    environment.slurm=True environment.resume=False \



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=after_rebut/sem_\${data.index}_\${data.len}_\${training.w_diffuse} \
    training=diffuse training.w_diffuse=0,0.01 \
    data=ho3d  data.index=SM2_0001_dt10,SMu1_0650_dt10,MDF10_1000_dt10 \
    environment.slurm=False environment.resume=False \



CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=ndc_intr/sds\${data.index}_\${novel_view.mode}_\${training.w_diffuse}_c\${training.clip} \
    training=diffuse training.w_diffuse=1e-3,1e-4,1e-2 training.clip=100,10  \
    data=ho3d  data.len=2 \
    environment.slurm=True environment.resume=False logging.mode=none \




# only with occluded sematnics 
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=ndc_intr_more/\${data.index}_\${novel_view.mode}_\${training.w_diffuse} \
    training=diffuse   \
    data=ho3d data.index=SM2_0001_dt02,MDF10_1000_dt02,SMu1_0650_dt02,SS2_0000_dt02 data.len=2 \
    environment.slurm=True environment.resume=False \


# with amodal normal and mask. 
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=ndc_intr_geom/\${data.index}_\${novel_view.mode}_d\${novel_view.loss.w_depth}n\${novel_view.loss.w_normal}x\${training.w_diffuse} \
    novel_view=geom novel_view.loss.w_depth=0,1 novel_view.loss.w_normal=0 \
    training=diffuse \
    data=ho3d  data.len=2 \
    environment.slurm=True environment.resume=False \




CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m train -m --config-name volsdf_nogt \
    expname=dev/tmp2 \
    training=diffuse training.w_diffuse=1 \
    data=ho3d data.len=2 \
    logging.mode=none \
    environment.slurm=False environment.resume=False \





CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=calib/more_\${data.index}_clip100_rescale_len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg_eik_\${training.w_eikonal} \
    training=diffuse training.w_diffuse=1,1e-2,10 \
    data=ho3d data.index=MDF10_1000_dt02,SMu1_0650_dt02,SS2_0000_dt02 data.len=2 \
    training.i_val=200 training.num_iters=2000 \
    environment.slurm=True environment.resume=False \




CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=calib/more_\${data.index}_clip100_rescale_len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg_eik_\${training.w_eikonal} \
    training=diffuse training.w_diffuse=1,1e-2,10 \
    data=ho3d data.index=MDF10_1000_dt02,SMu1_0650_dt02,SS2_0000_dt02 data.len=2 \
    training.i_val=200 training.num_iters=2000 \
    environment.slurm=True environment.resume=False \





CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=calib/one_rescale_len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg_eik_\${training.w_eikonal} \
    training=diffuse training.w_diffuse=1,1e-2,10 \
    data=ho3d data.index=SM2_0001_dt02 data.len=2 training.diff_name=ddpm\/glide_SM2\
    training.i_val=200 training.num_iters=2000 \
    environment.slurm=False environment.resume=False \



CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=calib/clip100_rescale_len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg_eik_\${training.w_eikonal} \
    training=diffuse training.w_diffuse=1,1e-2,10 \
    data=ho3d data.index=SM2_0001_dt02 data.len=2 \
    training.i_val=200 training.num_iters=2000 \
    environment.slurm=True environment.resume=False \


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=calib/\${data.index}_len\${data.len}_default  \
    data=ho3d data.index=SM2_0001_dt02 data.len=20,30 \
    training.i_val=200 training.num_iters=2000 \
    environment.slurm=True environment.resume=False \


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=calib/len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg training=diffuse training.w_diffuse=0,1e-4,1 \
    data=ho3d data.index=SM2_0001_dt02 data.len=2 \
    training.i_val=200 training.num_iters=2000 training.w_eikonal=0 \
    environment.slurm=True environment.resume=False \


-
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=fishrun/len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg training=diffuse training.w_diffuse=0.1,0.01,0.001 \
    data=ho3d data.index=SM2_0000_dt02 data.len=2 \
    training.i_val=200 \
    environment.slurm=True environment.resume=False \


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=fishrun/len\${data.len}_wdiff\${training.w_diffuse}_glide_train_seg training=diffuse training.w_diffuse=0,1 \
    data=ho3d data.index=SM2_0000_dt02 data.len=2,3 \
    training.i_val=200 \
    environment.slurm=True environment.resume=False \


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=fishrun/len\${data.len}_wdiff\${training.w_diffuse}_glide_one training=diffuse training.w_diffuse=0,1 \
    data=ho3d data.index=SM2_0000_dt02 data.len=2,3 \
    training.i_val=200 training.diff_name=ddpm\/glide_SM2 \
    environment.slurm=True environment.resume=False \




CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=w_diff/len\${data.len}_default  \
    data=ho3d data.index=SM2_0000_dt02 data.len=3 \
    training.i_val=200 \
    environment.slurm=True 

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=w_diff/len\${data.len}_default\${training.render_full_frame}  \
    data=ho3d data.index=SM2_0000_dt02 training.render_full_frame=True,False \
    training.i_val=200 \
    environment.slurm=True environment.resume=False \


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    data=ihoi_mow \
    expname=occ_ft/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    training.ckpt_file=\${output}/occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv}/lightning_logs/version_0/checkpoints/last.ckpt




CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=coord deepsdf.pe=True deepsdf.inv=False

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=coord deepsdf.pe=False deepsdf.inv=False


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=True deepsdf.inv=False

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=False deepsdf.inv=False


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=True deepsdf.inv=True

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${mode}_\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=False deepsdf.inv=True


--    


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=coord deepsdf.pe=True deepsdf.inv=False

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=coord deepsdf.pe=False deepsdf.inv=False


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=True deepsdf.inv=False

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=False deepsdf.inv=False


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=True deepsdf.inv=True

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name blind_prior \
    environment.multiprocessing_distributed=False \
    expname=occ/\${deepsdf.inp_mode}_\${deepsdf.pe}_\${deepsdf.inv} \
    deepsdf.inp_mode=dist deepsdf.pe=False deepsdf.inv=True


--

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_shapenet \
    expname=vox/lr_\${lr} \
    lr=1e-5


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_shapenet \
    expname=vox/lr_\${lr} \
    lr=2e-5

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_shapenet \
    expname=vox/lr_\${lr} \
    lr=1e-4


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name deep_sdf \
    expname=dev/tmp \
    environment.multiprocessing_distributed=True \
    deepsdf=art_cond data_mode=sdfhand train_split=train_mode frame=nSdf logging.mode=none



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name deep_sdf \
    expname=sdf/\${frame} \
    environment.multiprocessing_distributed=True \


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_shapenet \
    expname=canonical/vanila \
    environment.multiprocessing_distributed=True \

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_shapenet \
    expname=canonical/vanila_lr\${lr} \
    environment.multiprocessing_distributed=True \
    lr=1e-4


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_shapenet \
    expname=canonical/vanila_lr\${lr} \
    environment.multiprocessing_distributed=True \
    lr=1e-3





CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=transformer_art1224/coord\${unet_config.params.use_coord}_pe\${unet_config.params.pe_inp} \
    unet_config=art_attn batch_size=12 \
    unet_config.params.use_coord=True unet_config.params.pe_inp=True \

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=transformer_art1224/coord\${unet_config.params.use_coord}_pe\${unet_config.params.pe_inp} \
    unet_config=art_attn batch_size=12\
    unet_config.params.use_coord=False unet_config.params.pe_inp=False \

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=transformer_art1224/coord\${unet_config.params.use_coord}_pe\${unet_config.params.pe_inp} \
    unet_config=art_attn batch_size=12\
    unet_config.params.use_coord=False unet_config.params.pe_inp=True \

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=transformer_art1224/coord\${unet_config.params.use_coord}_pe\${unet_config.params.pe_inp} \
    unet_config=art_attn batch_size=12\
    unet_config.params.use_coord=True unet_config.params.pe_inp=False \

-
-
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=film/\${unet_config.mode}_p\${cf_prob} \
    unet_config=film_embed \
    environment.multiprocessing_distributed=True \
    unet_config.params.use_scale_shift_norm=True \



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=film/\${unet_config.mode}_p\${cf_prob} \
    unet_config=art_embed \
    environment.multiprocessing_distributed=True \
    unet_config.params.use_scale_shift_norm=True \




CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=film/\${unet_config.mode}_p\${cf_prob} \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True \
    unet_config.params.use_scale_shift_norm=True \



-
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=dev/\${unet_config.mode} \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True logging.mode=none \
    
-
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode/\${unet_config.mode}_p\${cf_prob} \
    unet_config=art_zero \
    environment.multiprocessing_distributed=True \
    cf_prob=0. \


--


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=film_embed \
    environment.multiprocessing_distributed=True \
    cf_prob=0.5 \



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=art_embed \
    environment.multiprocessing_distributed=True \
    cf_prob=0.5



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True \
    cf_prob=0.5


-

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=film_embed \
    environment.multiprocessing_distributed=True \
    cf_prob=0.1


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=art_embed \
    environment.multiprocessing_distributed=True \
    cf_prob=0.1



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True \
    cf_prob=0.1
-

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=film_embed \
    environment.multiprocessing_distributed=True \
    cf_prob=0.


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=art_embed \
    environment.multiprocessing_distributed=True \
    cf_prob=0.



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm_2mode \
    expname=2mode_scratch/\${unet_config.mode}_p\${cf_prob} \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True \
    cf_prob=0.

====






python -m ddpm.test  -m  \
    T=1,100,200,500,700 idx=0,1,2,3,4,5,6,7 \
    common_dir=/glusterfs/yufeiy2/vhoi/output_ddpm/art/ \
    data_dir=/glusterfs/yufeiy2/ihoi ## TODO
    ckpt=art/ckpt/model-00025000.pt,attention/ckpt/model-00025000.pt,film/ckpt/model-00025000.pt



python -m ddpm.test  -m  \
    T=1,100,200,500,700 idx=0,1,2,3,4,5,6,7 \
    common_dir=/glusterfs/yufeiy2/vhoi/output_ddpm/art/ \
    q_sample=True,False \
    ckpt=art/ckpt/model-00025000.pt,attention/ckpt/model-00025000.pt,film/ckpt/model-00025000.pt

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=dev/tmp3 \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True    

--
python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/art/art/ckpt/model-00020000.pt \
    --T 300 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/  \
    --q_sample 



CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=art/\${unet_config.mode} \
    unet_config=ddim_cond \
    environment.multiprocessing_distributed=True


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=art/\${unet_config.mode} \
    unet_config=film_embed \
    environment.multiprocessing_distributed=True




CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=art/\${unet_config.mode} \
    unet_config=art_embed \
    environment.multiprocessing_distributed=True



python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00017500.pt \
    --T 300 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/ --split train_full
    --q_sample 

python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00017500.pt \
    --T 300 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/   --split train_full


python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00017500.pt \
    --T 100 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/  --split train_full
    --q_sample

python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00017500.pt \
    --T 100 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/ --split train_full



python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00017500.pt \
    --T 500 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/ --split train_full
    --q_sample

python -m ddpm.test \
    --ckpt /glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00017500.pt \
    --T 500 \
    --data_dir /glusterfs/yufeiy2/vhoi/mow/ --split train_full




PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=mow/\${data.index}_\${training.w_diffuse}_100_1000 \
    training=diffuse  \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_diffuse=0. \
    training.diffuse_ckpt=/glusterfs/yufeiy2/vhoi/output_ddpm/mow/t100_1000_condTrue/ckpt/model-00015000.pt \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=mow/\${data.index}_\${training.w_diffuse}_100_1000 \
    training=diffuse  \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_diffuse=10. \
    training.diffuse_ckpt=/glusterfs/yufeiy2/vhoi/output_ddpm/mow/t100_1000_condTrue/ckpt/model-00015000.pt \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=mow/\${data.index}_\${training.w_diffuse}_1000_1000 \
    training=diffuse  \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_diffuse=10. \
    training.diffuse_ckpt=/glusterfs/yufeiy2/vhoi/output_ddpm/mow/t1000_1000_condTrue/ckpt/model-00015000.pt \
    device_ids=[0,1]



PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=mow/\${data.index}_\${training.w_diffuse}_100_100 \
    training=diffuse  \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_diffuse=10. \
    training.diffuse_ckpt=/glusterfs/yufeiy2/vhoi/output_ddpm/mow/t100_100_condTrue/ckpt/model-00015000.pt \
    device_ids=[0,1]




-
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=mow/t\${time.start}_\${time.total}_cond\${unet_config.params.use_spatial_transformer} \
    time.start=100 time.total=100 \
    data_dir=/glusterfs/yufeiy2/vhoi/mow/ \
    environment.multiprocessing_distributed=True

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=mow/t\${time.start}_\${time.total}_cond\${unet_config.params.use_spatial_transformer} \
    time.start=100 time.total=1000 \
    data_dir=/glusterfs/yufeiy2/vhoi/mow/ \
    environment.multiprocessing_distributed=True

    CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name ddpm \
        expname=mow/t\${time.start}_\${time.total}_cond\${unet_config.params.use_spatial_transformer} \
        time.start=1000 time.total=1000 \
        data_dir=/glusterfs/yufeiy2/vhoi/mow/ \
        environment.multiprocessing_distributed=True


--
contact loss

diffusion loss

PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=try_diffuse/\${data.index}_\${training.w_diffuse} \
    training=diffuse  \
    data.index=packing_v__VKclLReM0Y_frame000352_0 \
    training.w_diffuse=0. \
    device_ids=[0,1]



PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=try_diffuse/\${data.index}_\${training.w_diffuse} \
    training=diffuse  \
    training.w_diffuse=0. \
    device_ids=[0,1]



PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=try_diffuse/\${data.index}_\${training.w_diffuse} \
    training=diffuse  \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_diffuse=10. \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=try_diffuse/\${data.index}_\${training.w_diffuse} \
    training=diffuse  \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=try_diffuse/\${data.index}_\${training.w_diffuse} \
    training=diffuse  \
    data.index=packing_v__VKclLReM0Y_frame000352_0 \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=try_diffuse/\${data.index}_\${training.w_diffuse} \
    training=diffuse  \
    data.index=study_v_im0FA2X6fp0_frame000043_0 \
    device_ids=[0,1]


no loss



--
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    time.start=100 time.total=1000 \
    unet_config=ddim_uncond  \
    environment.multiprocessing_distributed=True 

--


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=condition/t\${time.start}_\${time.total}_cond\${unet_config.params.use_spatial_transformer} \
    time.start=100 time.total=1000 \
    unet_config=ddim_uncond  \
    environment.multiprocessing_distributed=True

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=condition/t\${time.start}_\${time.total}_cond\${unet_config.params.use_spatial_transformer} \
    time.start=100 time.total=100 \
    unet_config=ddim_uncond  \
    environment.multiprocessing_distributed=True


CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=condition/t\${time.start}_\${time.total}_cond\${unet_config.params.use_spatial_transformer} \
    time.start=1000 time.total=1000 \
    unet_config=ddim_uncond  \
    environment.multiprocessing_distributed=True

--

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=condition/t\${time.start}_\${time.total} \
    time.start=100 time.total=1000 \
    environment.multiprocessing_distributed=True

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=condition/t\${time.start}_\${time.total} \
    time.start=100 time.total=100 \
    environment.multiprocessing_distributed=True


CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m engine --config-name ddpm \
    expname=condition/t\${time.start}_\${time.total} \
    time.start=1000 time.total=1000 \
    environment.multiprocessing_distributed=True




PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=tmp \


--100doh

python -m ddpm.sdf_data --skip --gpu 0 --split all & 
python -m ddpm.sdf_data --skip --gpu 0 --split all & 

python -m ddpm.sdf_data --skip --gpu 1 --split all & 
python -m ddpm.sdf_data --skip --gpu 1 --split all & 

python -m ddpm.sdf_data --skip --gpu 2 --split all & 
python -m ddpm.sdf_data --skip --gpu 2 --split all & 

python -m ddpm.sdf_data --skip --gpu 3 --split all & 
python -m ddpm.sdf_data --skip --gpu 3 --split all & 

# together 
PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    training=contact  \
    device_ids=[0,1]

PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    data.index=packing_v__VKclLReM0Y_frame000352_0 \
    training=contact  \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    data.index=diy_v_Gwv7L53aONY_frame000080_0 \
    training=contact  \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    data.index=diy_v_e-6TZF3jCDk_frame000089_1 \
    training=contact  \
    device_ids=[0,1]




PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact training.w_contour=10 \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact_contour/contact_\${training.w_contact}_contour\${training.w_contour}_lr\${training.lr.pose} \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact \
    training.lr.pose=5e-3 \
    device_ids=[0,1]


-    


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contour/w_\${training.w_contour} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contour/w_\${training.w_contour} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_contour=10. \
    device_ids=[2,3]




PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contour/w_\${training.w_contour} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_contour=0. \
    device_ids=[0,1]

PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contour/w_\${training.w_contour} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.w_contour=1000. \
    device_ids=[2,3]


--
PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=debug_hand/\${training.w_contact} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training.occ_mask=indp training.num_iters=10000 \
    device_ids=[0,1]


-

PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact/\${training.w_contact}_\${training.backward} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact training.w_contact=1. training.backward='pose' training.w_contour=0 \
    device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact/\${training.w_contact}_\${training.backward} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact training.w_contact=0.1 training.backward='pose' training.w_contour=0 \
    device_ids=[0,1]




PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact/\${training.w_contact}_\${training.backward} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact training.w_contact=1. training.backward='once' \
    device_ids=[0,1]




PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact/\${training.w_contact}_\${training.backward} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact training.w_contact=0.1 \
    device_ids=[2,3]



PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=contact/\${training.w_contact}_\${training.backward} data.index=study_v_fFyBlNmK1N8_frame000411_0 \
    training=contact training.w_contact=0.1 training.backward='once' \
    device_ids=[0,1]

---c

PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=smooth_cam/\${data.index} \
    data.index=packing_v__VKclLReM0Y_frame000352_0 device_ids=[0,1]
    
PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=smooth_cam/\${data.index} \
    data.index=diy_v_Gwv7L53aONY_frame000080_0 device_ids=[2,3]

PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=smooth_cam/\${data.index} \
    data.index=diy_v_e-6TZF3jCDk_frame000089_1 device_ids=[0,1]


PYTHONPATH=. python -m engine --config-name volsdf_nogt \
    expname=smooth_cam/\${data.index} \
    data.index=study_v_fFyBlNmK1N8_frame000411_0 device_ids=[2,3]



---
PYTHONPATH=. python -m engine \
    expname=smooth/cam\${camera.mode}_oTh\${oTh.mode}_hA\${hA.mode}\${training.w_t_hand}_f\${data.focal} \
    data=100doh camera=delta oTh=learn hA.mode=learn training.w_t_hand=100. 



PYTHONPATH=. python -m engine \
    expname=smooth/cam\${camera.mode}_oTh\${oTh.mode}_hA\${hA.mode}\${training.w_t_hand} \
    data=100doh camera=delta oTh=learn hA.mode=learn training.w_t_hand=100.



PYTHONPATH=. python -m engine \
    expname=smooth/cam\${camera.mode}_oTh\${oTh.mode}_hA\${hA.mode}\${training.w_t_hand} \
    data=100doh camera=delta oTh=learn hA.mode=learn training.w_t_hand=10.


PYTHONPATH=. python -m engine \
    expname=smooth/cam\${camera.mode}_oTh\${oTh.mode}_hA\${hA.mode}\${training.w_t_hand} \
    data=100doh camera=delta oTh=learn hA.mode=learn training.w_t_hand=1.



PYTHONPATH=. python -m engine \
    expname=100doh/cam\${camera.mode}_oTh\${oTh.mode} \
    data=100doh camera=delta oTh=learn

PYTHONPATH=. python -m engine \
    expname=100doh/cam\${camera.mode}_oTh\${oTh.mode} \
    data=100doh camera=gt oTh=gt \


PYTHONPATH=. python -m engine \
    expname=100doh/cam\${camera.mode}_oTh\${oTh.mode} \
    data=100doh camera=gt oTh=learn \

PYTHONPATH=. python -m engine \
    expname=100doh/cam\${camera.mode}_oTh\${oTh.mode} \
    data=100doh camera=learn oTh=gt

PYTHONPATH=. python -m engine \
    expname=100doh/cam\${camera.mode}_oTh\${oTh.mode} \
    data=100doh camera=gt oTh=learn



PYTHONPATH=. python -m engine \
    environment=dev \
    data=100doh


PYTHONPATH=. python -m engine \
    environment=dev \
    expname=art/hA_\${hA.mode}_\${training.w_sdf} \
    data.index=SMu1_0650


--
# optimize hand articulation

PYTHONPATH=. python -m engine -m \
    expname=art/hA_\${hA.mode}_\${training.w_sdf} \
    hA.mode=learn,gt training.w_sdf=0.01,0 


--


PYTHONPATH=. python -m engine -m \
    expname=honey_grow/hand_text_\${data.index}_\${oTh.mode}_wm\${training.w_mask} \
    oTh=gt,learn \
    data.index=BB12_0000,AP12_0050,MDF10_0000,SMu41_0000,SS2_0000 \
    training.w_mask=1,10


MC2_0000,GSF11_1000,SM2_0000


--    
PYTHONPATH=. python -m engine -m \
    expname=honey_grow/hand_text_\${data.index}_\${oTh.mode} \
    oTh=gt,learn \
    data.index=MC2_0000,GSF11_1000,SM2_0000


PYTHONPATH=. python -m engine -m \
    expname=honey_grow/hand_text_\${data.index}_\${oTh.mode}_wm\${training.w_mask} \
    oTh=gt,learn \
    data.index=MC2_0000,GSF11_1000,SM2_0000 \
    training.w_mask=1


# bu 
PYTHONPATH=. python -m engine -m \
    expname=honey_grow/hand_text_\${data.index}_\${oTh.mode}_wm\${training.w_mask} \
    oTh=learn \
    data.index=SM2_0000 \
    training.w_mask=50




# separation


---
PYTHONPATH=. python -m engine \
    expname=dev/hand_text \
    environment=dev 




PYTHONPATH=. python -m engine -m \
    expname=render_label2_rgb/\${blend_train.method}_order\${training.label_prob} \
    blend_train.method=vol,soft,hard training.occ_mask=label training.label_prob=1 \


PYTHONPATH=. python -m engine -m \
    expname=render_label2_rgb/\${blend_train.method}_order\${training.label_prob} \
    blend_train.method=soft training.occ_mask=label training.label_prob=2 \



---
PYTHONPATH=. python -m engine -m \
    expname=render_label2/\${blend_train.method}_order\${training.label_prob} \
    blend_train.method=vol,soft,hard training.occ_mask=label training.label_prob=1 \
    training.w_eikonal=0 training.w_rgb=0 

PYTHONPATH=. python -m engine -m \
    expname=render_label2/\${blend_train.method}_order\${training.label_prob} \
    blend_train.method=soft training.occ_mask=label training.label_prob=2 \
    training.w_eikonal=0 training.w_rgb=0 environment=dev


--
PYTHONPATH=. python -m engine -m \
    expname=dev/order\${training.label_prob}_wd\${training.w_depth} \
    blend_train.method=vol training.occ_mask=label training.label_prob=1 \
    training.w_eikonal=0 training.w_rgb=0  environment=dev



PYTHONPATH=. python -m engine -m \
    expname=render_label/order\${training.label_prob}_wd\${training.w_depth} \
    blend_train.method=soft training.occ_mask=label training.label_prob=1,2 training.w_depth=1.0,0.0 \
    training.w_eikonal=0 training.w_rgb=0 

PYTHONPATH=. python -m engine -m \
    expname=render_label/\${blend_train.method}_\${training.occ_mask}_\${training.label_prob} \
    blend_train.method=soft training.occ_mask=label training.label_prob=1 training.w_depth 1.0 \
    training.w_eikonal=0 training.w_rgb=0 environment=dev


--


PYTHONPATH=. python -m engine \
    expname=test_requeue/\${data.index}_\${blend_train.method}_\${training.occ_mask} \
    data.index=AP12_0050,SM2_0000 \
    blend_train.method=soft training.occ_mask=label \
    environment=learn \


# only mask, make sure gradient works? 
PYTHONPATH=. python -m engine -m \
    expname=only_mask/\${blend_train.method}_\${training.occ_mask} \
    blend_train.method=soft,hard training.occ_mask=indp,union,label \
    training.w_eikonal=0 training.w_rgb=0 

# see soft or hard blending~~
PYTHONPATH=. python -m engine -m \
    expname=soft_hard/\${blend_train.method}_\${training.occ_mask} \
    blend_train.method=soft,hard training.occ_mask=indp,union,label \



--
PYTHONPATH=. python -m engine \
    expname=dev_blend/\${blend_train.method} \
    environment=dev \
    blend_train.method=soft


PYTHONPATH=. python -m engine \
    expname=dev_blend/\${blend_train.method} \
    training.monitoring=none \
    blend_train.method=hard



DATA=AP12_0050
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/unknown_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --oTh:learn_R 1 --oTh:learn_t 1 --oTh:mode learn \
    --slurm --sl_ngpu 2 


DATA=AP12_0050
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/gt_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --slurm --sl_ngpu 2 




-
DATA=SMu1_0650
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/unknown_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --oTh:learn_R 1 --oTh:learn_t 1 --oTh:mode learn \
    --slurm --sl_ngpu 2 


DATA=SMu1_0650
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/gt_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --slurm --sl_ngpu 2 






DATA=MDF10_0000
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/unknown_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --oTh:learn_R 1 --oTh:learn_t 1 --oTh:mode learn \
    --slurm --sl_ngpu 2 


DATA=MDF10_0000
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/gt_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --slurm --sl_ngpu 2 




DATA=MDF10_0090
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/unknown_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --oTh:learn_R 1 --oTh:learn_t 1 --oTh:mode learn \
    --slurm --sl_ngpu 2 


DATA=MDF10_0090
python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/gt_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --slurm --sl_ngpu 2 



in objnorm coord
python -m train --config configs/volsdf_hoi.yaml \
    --expname ho3d_known_cam/MDF10_0090 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/MDF10_0090 \
    --slurm --sl_ngpu 2 

python -m train --config configs/volsdf_hoi.yaml \
    --expname ho3d_known_cam/MDF10_0090_hand --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/MDF10_0090 \
    --model:joint_frame hand_norm \
    --slurm --sl_ngpu 2 



python -m train --config configs/volsdf_hoi.yaml \
    --expname ho3d_known_cam/MDF10_0000 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/MDF10_0000 \
    --slurm --sl_ngpu 2 

python -m train --config configs/volsdf_hoi.yaml \
    --expname ho3d_known_cam/MDF10_0000_hand --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/MDF10_0000 \
    --model:joint_frame hand_norm \
    --slurm --sl_ngpu 2 



python -m train --config configs/volsdf_hoi.yaml \
    --expname ho3d_known_cam/SMu1_0650 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/SMu1_0650 \
    --slurm --sl_ngpu 2 

python -m train --config configs/volsdf_hoi.yaml \
    --expname ho3d_known_cam/SMu1_0650_hand --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/SMu1_0650 \
    --model:joint_frame hand_norm \
    --slurm --sl_ngpu 2 




-
python -m train --config configs/volsdf_hoi.yaml \
    --expname dev/MDF10_0090 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/MDF10_0090 \
 \




python -m train --config configs/volsdf_hoi.yaml \
    --expname scale_radius/-1 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:type HOI_dtu \
    --slurm --sl_ngpu 2 

python -m train --config configs/volsdf_hoi.yaml \
    --expname scale_radius/3 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius 3 --data:type HOI_dtu \
    --slurm --sl_ngpu 2 


-
python -m train --config configs/volsdf_hoi.yaml \
    --expname debug/sdf_0.01 --training:occ_mask indp  \
    --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 --training:w_sdf 0.01 \


--=
python -m train --config configs/volsdf.yaml --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/00006755 \
    --expname dev/agnostic_blue  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \
    --slurm --ddp


python -m train --config configs/volsdf_hoi.yaml \
    --expname depth/sdf_no --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0. \
    --slurm --sl_ngpu 2 

python -m train --config configs/volsdf_hoi.yaml \
    --expname depth/sdf_0.01 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 \
    --slurm --sl_ngpu 2 

python -m train --config configs/volsdf_hoi.yaml \
    --expname depth/sdf_0.1 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.1 \
    --slurm --sl_ngpu 2 


python -m train --config configs/volsdf_hoi.yaml \
    --expname depth/sdf_1 --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 1 \
    --slurm --sl_ngpu 2 





python -m train --config configs/volsdf_hoi.yaml \
    --expname dev/indp --training:occ_mask indp  \
    --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \



python -m train --config configs/volsdf_hoi.yaml \
    --expname occ/union --training:occ_mask union  \
    --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \
    --slurm --ddp

python -m train --config configs/volsdf_hoi.yaml \
    --expname occ/indp --training:occ_mask indp  \
    --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 --test_train 1 \
    --slurm --ddp


-
python -m train --config configs/volsdf_hoi.yaml \
    --expname cmp/hybrid_blue  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/00006755 \
    --expname cmp/agnostic_blue  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \
    --slurm --ddp

--

python -m train --config configs/volsdf_hoi.yaml \
    --expname dev/rgb_blue  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \


--
python -m train --config configs/volsdf.yaml \
    --expname syn/rgb_dtu  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml \
    --expname syn/flow_dtu  --training:w_mask 1.0 --training:w_flow 1.0 --training:fg 1 \
    --slurm --ddp


-


python -m train --config configs/volsdf.yaml  --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/00006755 \
    --expname syn/rgb_blue  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 1 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/00006755 \
    --expname syn/flow_blue  --training:w_mask 1.0 --training:w_flow 1.0 --training:fg 1 \
    --slurm --ddp

---

python -m train --config configs/volsdf.yaml  \
    --expname right_mask/mask_rgb_cam  --training:w_mask 1.0 --training:w_flow 0.0 \
    --camera:mode para --training:num_iters 500000 \
    --training:i_val 2000 --training:i_val_mesh 2000 --training:i_save 2000 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname right_mask/mask_flow_cam  --training:w_mask 1.0 --training:w_flow 1.0 \
    --camera:mode para --training:num_iters 500000 \
    --training:i_val 2000 --training:i_val_mesh 2000 --training:i_save 2000 \
    --slurm --ddp        




--
python -m train --config configs/volsdf.yaml  \
    --expname right_mask/mask_rgb_bg  --training:w_mask 1.0 --training:w_flow 0.0 --training:fg 0 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname right_mask/mask_flow_bg  --training:w_mask 1.0 --training:w_flow 1.0 --training:fg 0 \
    --slurm --ddp        


python -m train --config configs/volsdf.yaml  \
    --expname right_mask/mask_rgb  --training:w_mask 1.0 --training:w_flow 0.0 \
    --slurm --ddp


python -m train --config configs/volsdf.yaml  \
    --expname right_mask/mask_flow  --training:w_mask 1.0 --training:w_flow 1.0 \
    --slurm --ddp        


-
python -m train --config configs/volsdf.yaml  \
    --expname right_mask/rgb_bg  --training:w_mask 0.0 --training:w_flow 0.0 --training:fg 0 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname right_mask/flow_bg  --training:w_mask 0.0 --training:w_flow 1.0 --training:fg 0 \
    --slurm --ddp        


python -m train --config configs/volsdf.yaml  \
    --expname right_mask/rgb  --training:w_mask 0.0 --training:w_flow 0.0 \
    --slurm --ddp


python -m train --config configs/volsdf.yaml  \
    --expname right_mask/flow  --training:w_mask 0.0 --training:w_flow 1.0 \
    --slurm --ddp        



---
python -m train --config configs/volsdf.yaml  \
    --expname gt_rightflow/rgb_bg  --training:w_mask 0.0 --training:w_flow 0.0 --training:fg 0 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname gt_rightflow/flow_bg  --training:w_mask 0.0 --training:w_flow 1.0 --training:fg 0 \
    --slurm --ddp        


python -m train --config configs/volsdf.yaml  \
    --expname gt_rightflow/rgb  --training:w_mask 0.0 --training:w_flow 0.0 \
    --slurm --ddp


python -m train --config configs/volsdf.yaml  \
    --expname gt_rightflow/flow  --training:w_mask 0.0 --training:w_flow 1.0 \
    --slurm --ddp        

--
python -m train --config configs/volsdf.yaml  \
    --expname paracam/rgb_bg  --training:w_mask 0.0 --training:w_flow 0.0 --training:fg 0 \
    --camera:mode para \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname paracam/flow_bg  --training:w_mask 0.0 --training:w_flow 1.0 --training:fg 0 \
    --camera:mode para \
    --slurm --ddp        


python -m train --config configs/volsdf.yaml  \
    --expname paracam/rgb  --training:w_mask 0.0 --training:w_flow 0.0 \
    --camera:mode para \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname paracam/mask  --training:w_mask 1.0 --training:w_flow 0.0 \
    --camera:mode para \
    --slurm --ddp    

python -m train --config configs/volsdf.yaml  \
    --expname paracam/flow  --training:w_mask 0.0 --training:w_flow 1.0 \
    --camera:mode para \
    --slurm --ddp        

python -m train --config configs/volsdf.yaml  \
    --expname paracam/flow_mask  --training:w_mask 1.0 --training:w_flow 1.0 \
    --camera:mode para \
    --slurm --ddp            


-
python -m train --config configs/volsdf.yaml  \
    --expname gt/rgb_bg  --training:w_mask 0.0 --training:w_flow 0.0 --training:fg 0 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname gt/flow_bg  --training:w_mask 0.0 --training:w_flow 1.0 --training:fg 0 \
    --slurm --ddp        


python -m train --config configs/volsdf.yaml  \
    --expname gt/rgb  --training:w_mask 0.0 --training:w_flow 0.0 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname gt/mask  --training:w_mask 1.0 --training:w_flow 0.0 \
    --slurm --ddp    

python -m train --config configs/volsdf.yaml  \
    --expname gt/flow  --training:w_mask 0.0 --training:w_flow 1.0 \
    --slurm --ddp        

python -m train --config configs/volsdf.yaml  \
    --expname gt/flow_mask  --training:w_mask 1.0 --training:w_flow 1.0 \
    --slurm --ddp            





----
python -m train --config configs/volsdf.yaml  \
    --expname sushi/rgb  --training:w_mask 0.0 --training:w_flow 0.0 \
    --slurm --ddp

python -m train --config configs/volsdf.yaml  \
    --expname sushi/mask  --training:w_mask 1.0 --training:w_flow 0.0 \
    --slurm --ddp    

python -m train --config configs/volsdf.yaml  \
    --expname sushi/flow  --training:w_mask 0.0 --training:w_flow 1.0 \
    --slurm --ddp        

python -m train --config configs/volsdf.yaml  \
    --expname sushi/flow_mask  --training:w_mask 1.0 --training:w_flow 1.0 \
    --slurm --ddp            