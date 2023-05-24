

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=rebuttal/\${data.index}_\${suf}  \
    suf='_smooth_100_x150_x150,_smooth_100_x50_x50,_smooth_100_x200_x200' \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1  \
    training.w_rgb=0 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=rebuttal/\${data.index}_\${suf}  \
    suf='' \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1  \
    training.w_rgb=0 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=rebuttal/\${data.index}_\${suf}  \
    suf='_smooth_100_pred_x200,_smooth_100_x200_pred,_smooth_100_pred_x150,_smooth_100_x150_pred' \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1  \
    training.w_rgb=0 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &


-
run all fig
PYTHONPATH=. python -m tools.vis_clips  -m     load_folder=which_prior_w0.01_exp/,pred_no_prior/,ablate_weight/,which_prior_w0.01/

-
PYTHONPATH=. python -m tools.vis_clips  -m \
    load_folder=hhor_less/ \

PYTHONPATH=. python -m tools.vis_clips  -m \
    load_folder=hhor_less,hhor_less_w0.0001_1e-05,hhor_less_w0.001_1e-05,hhor_less_w0.01_1e-05,hhor_less_w1e-05_1e-05 \
    hydra/launcher=slurm \

PYTHONPATH=. python -m tools.vis_clips  -m \
    load_folder=hhor_less/6_,hhor_less_w0.0001_1e-05/6_,hhor_less_w0.001_1e-05/6_,hhor_less_w0.01_1e-05/6_ \
    hydra/launcher=slurm \

Mug,Bottle,Kettle,Bowl,Knife,ToyCar

[in the wild]


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=wild_gray/\${data.name}\${data.index}  \
    training.w_rgb=0 \
    data=custom data.name='1st_nocrop' data.index=bottle_1,bottle_2,mug_3,mug_1,kettle_4,kettl_2,kettle_5,knife_3,bowl_2,bowl_4,bowl_1 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=wild_gray/\${data.name}\${data.index}  \
    training.w_rgb=0 \
    data=custom data.name='1st_nocrop' data.index=bottle_2 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=wild_gray/\${data.name}\${data.index}  \
    training.w_rgb=0 \
    data=custom data.name='3rd_nocrop' data.index=kettle6,knife1,bowl2,mug1,bottle2,bowl1,knife2,knife3,mug3,mug2,knife6,bottle6,kettle4 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360




CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=wild_gray/\${data.name}\${data.index}  \
    data=custom data.index=Kettle_101,Kettle_102 \
    novel_view.sd_para.anneal_noise=exp training.w_rgb=0 \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360


[not learn oTh]
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=gray_oTh/\${data.index}_suf\${suf}_\${oTh.learn_R}_\${oTh.learn_t} \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1 \
    oTh.learn_R=False oTh.learn_t=False \
    training.w_rgb=0 \
    hydra/launcher=slurm


[ordinal depth]
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=oridnal_depth/\${data.index}_suf\${suf}_depth\${training.w_depth} \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1 \
    training.w_rgb=0 training.w_depth=1 \
    hydra/launcher=slurm


[IMPORTANT ! MAIN EPX]
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=gray_which_prior_w\${training.w_diffuse}_\${novel_view.sd_para.anneal_noise}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1 \
    training.w_rgb=0 \
    novel_view.diff_index=ObjGeomGlide_cond_all_linear_catTrue_cfgFalse,CondGeomGlide_cond_all_linear_catFalse_cfgFalse \
    hydra/launcher=slurm

CondGeomGlide_cond_all_linear_catTrue_cfgFalse,

--
bu

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=which_prior2_w\${training.w_diffuse}_\${novel_view.sd_para.anneal_noise}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.index=Kettle_2 \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.diff_index=ObjGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=which_prior_w\${training.w_diffuse}_\${novel_view.sd_para.anneal_noise}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.index=Kettle_1 \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=which_prior_w\${training.w_diffuse}_\${novel_view.sd_para.anneal_noise}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.index=Bottle_2 \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catFalse_cfgFalse \
    hydra/launcher=slusrm


-
[hhor] 

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less_more_w\${training.w_diffuse}/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}_10k  \
    novel_view.sd_para.guidance_scale=0 suf='' \
    training.w_diffuse=1e-3 \
    data=hhor data.ratio=0.2,0.05 \
    data.index=28_Doraemon training.num_iters=10000 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less_more_w\${training.w_diffuse}/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}  \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.sd_para.guidance_scale=0 suf='' \
    training.w_diffuse=1e-3 \
    data=hhor data.ratio=0.9 \
    data.index=1_Orange,2_Plastic_Box,13_Bulbasaur,12_Jigglypuff,23_Yellow_Tiger,25_Remote_Control,28_Doraemon,29_Apple_Pencil \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360


1_Orange,2_Plastic_Box,13_Bulbasaur,12_Jigglypuff,23_Yellow_Tiger,25_Remote_Control,28_Doraemon,29_Apple_Pencil


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less_tiger_w\${training.w_diffuse}/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}  \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.sd_para.guidance_scale=0 suf='' \
    training.w_diffuse=1e-3,1e-2 \
    data=hhor data.ratio=0.05,0.2,0.5,0.9 \
    data.index=22_Red_Tiger \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360

-

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less_tiger_w\${training.w_diffuse}/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}  \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.sd_para.guidance_scale=0 suf='' \
    training.w_diffuse=1e-3,1e-2 \
    data=hhor data.ratio=0.05,0.5,0.9 \
    data.index=22_Red_Tiger \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less_tiger_w\${training.w_diffuse}/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}  \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.sd_para.guidance_scale=0 suf='' \
    training.w_diffuse=0 training.render_full_frame=False \
    data=hhor data.ratio=0.9 \
    data.index=22_Red_Tiger \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less_w\${training.w_diffuse}_\${training.lr.pose}/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}  \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.sd_para.guidance_scale=0 suf='' \
    training.w_diffuse=1e-3 \
    data=hhor data.ratio=0.05,0.1,0.15,0.2,0.25,0.5,1 training.lr.pose=1e-5 \
    data.index=6_AirPods \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360


-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=hhor_less/\${data.index}_\${data.offset}_\${data.ratio}_suf\${suf}  \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.sd_para.guidance_scale=0 suf='' \
    data=hhor data.ratio=0.05,0.1,0.15,0.2,0.25,0.5,1  \
    data.index=6_AirPods \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360


-
[ablation for annealing]
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=gray_which_prior_w\${training.w_diffuse}_\${novel_view.sd_para.anneal_noise}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1 novel_view.sd_para.anneal_noise=constant \
    training.w_rgb=0 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360



# zfar
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=which_prior_w\${training.w_diffuse}_\${novel_view.sd_para.anneal_noise}_zfar/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2 \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse_1_1 \
    hydra/launcher=slurm hydra.launcher.timeout_min=360

-
[ablation for weight]

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=ablate_color/\${data.index}_rgb\${training.w_rgb}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=2  \
    training.w_rgb=0 \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=ablate_weight_gray/\${data.index}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1  \
    novel_view.loss.w_mask=0 training.w_rgb=0  \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=ablate_weight_gray/\${data.index}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1  \
    novel_view.loss.w_normal=0 training.w_rgb=0  \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=ablate_weight_gray/\${data.index}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1  \
    novel_view.loss.w_depth=0 training.w_rgb=0  \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=ablate_weight_gray2/\${data.index}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth}  \
    data.cat=Knife data.ind=1  training.num_iters=10000 \
    novel_view.loss.w_depth=0 training.w_rgb=0  \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm hydra.launcher.timeout_min=360 &


-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=which_prior_w\${training.w_diffuse}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2 \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse,ObjGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm



# bu
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=which_prior_w\${training.w_diffuse}/\${data.index}_suf\${suf}_\${novel_view.diff_index}  \
    data.index=ToyCar_2 \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm


-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=anneal/\${data.index}_suf\${suf}_\${novel_view.sd_para.anneal_noise}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar \
    novel_view.sd_para.anneal_noise=exp \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm



-

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=anneal/\${data.index}_suf\${suf}_\${novel_view.sd_para.anneal_noise}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar \
    novel_view.sd_para.anneal_noise=sqrt,linear,exp,cosine \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse \
    hydra/launcher=slurm


# NO PRIOR!!!
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m \
    expname=pred_no_prior_gray/\${data.index}_suf\${suf} \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1 training.w_rgb=0 \
    training.render_full_frame=False training.w_diffuse=0 \
    hydra/launcher=slurm 







CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=pred/\${data.index}_len\${data.len}_suf\${suf}_lrpose\${training.lr.pose}xobj\${training.lr.oTh}_\${novel_view.diff_index}  \
    data.index=Knife_1_0,Knife_1_1,Bowl_1_0 suf='_smooth_100' \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse,CondGeomGlide_cond_all_linear_catFalse_cfgFalse,ObjGeomGlide_cond_all_linear_catTrue_cfgFalse \
    environment.slurm=True environment.exclude_nodes="grogu-1-9+grogu-1-24+grogu-2-9"  environment.resume=False logging.mode=none 


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=pred_go/\${data.index}_len\${data.len}_suf\${suf}_w\${training.w_diffuse}_lrpose\${training.lr.pose}xobj\${training.lr.oTh}_\${novel_view.diff_index}  \
    data.index=Bottle_1_0,Kettle_1_0,Knife_1_0,Knife_1_1,Mug_1_0,ToyCar_1_0,Bowl_1_0 suf='_smooth_100' \
    novel_view.diff_index=CondGeomGlide_cond_all_linear_catTrue_cfgFalse,CondGeomGlide_cond_all_linear_catFalse_cfgFalse,ObjGeomGlide_cond_all_linear_catTrue_cfgFalse \
    training.w_diffuse=1e-2 \
    environment.slurm=True environment.exclude_nodes="grogu-1-9+grogu-1-24+grogu-2-9"  environment.resume=False logging.mode=none 



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=dev/tmp \
    data.index=Bottle_1_0,Kettle_1_0,Knife_1_0,Knife_1_1,Mug_1_0,ToyCar_1_0,Bowl_1_0 suf='_smooth_100' \
    novel_view.diff_index=ObjGeomGlide_cond_all_linear_catTrue_cfgFalse \
    environment.slurm=False environment.exclude_nodes="grogu-1-9+grogu-1-24+grogu-2-9"  environment.resume=False logging.mode=none 


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=pred/\${data.index}_len\${data.len}_\${suf}_lrpose\${training.lr.pose}xobj\${training.lr.oTh}  \
    training.lr.pose=5e-4,1e-5 training.lr.oTh=5e-4,1e-5 \
    environment.slurm=False environment.slurm_timeout=120 environment.resume=False logging.mode=none 

-

-
python -m preprocess.clip_pred_hand --batch --inp /home/yufeiy2/scratch/result/HOI4D/ --skip 

-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=anneal/hoi4d_len\${data.len}_\${novel_view.sd_para.anneal_noise}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=1000  novel_view.sd_para.anneal_noise=sqrt,constant,linear,exp,cosine\
    training.warmup=100 pixel_sampler.name=naive \
    environment.slurm=True environment.slurm_timeout=480 environment.resume=False logging.mode=none 





CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=artifact/hoi4d_len\${data.len}_whand\${training.w_hand_mask}_lrpose\${training.lr.pose}xobj\${training.lr.oTh}_\${pixel_sampler.name}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=1000 training.lr.pose=5e-4,1e-5 training.lr.oTh=5e-4,1e-5 training.w_hand_mask=1,0,10 \
    training.warmup=100 pixel_sampler.name=proportion,naive \
    environment.slurm=False environment.slurm_timeout=120 environment.resume=False logging.mode=none 

-

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=artifact/hoi4d_len\${data.len}_lrpose\${training.lr.pose}_w\${training.w_diffuse}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=1000 training.lr.pose=1e-4,5e-4 \
    data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159 \
    training.warmup=100 novel_view.loss.w_schdl=bell \
    environment.slurm=False environment.resume=False logging.mode=none 

-
ddpm_novel_sunday/hoi4d_CondGeomGlide_1

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate_seq/obj_prior_len\${data.len}_w\${training.w_diffuse}_\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d data.len=1000 \
    data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259 \
    training.warmup=100,1000 novel_view.loss.w_schdl=bell \
    environment.slurm=False environment.resume=False logging.mode=none 




CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate_seq/obj_prior_len\${data.len}_w\${training.w_diffuse}_\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_catTrue_cfgFalse" \
    data=hoi4d data.len=1000 \
    data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259 \
    training.warmup=100,1000 novel_view.loss.w_schdl=bell \
    environment.slurm=False environment.resume=False logging.mode=none 


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate/obj_prior_len\${data.len}_w\${training.w_diffuse}_\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_catTrue_cfgFalse" \
    data=hoi4d data.len=2 \
    data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259 \
    training.warmup=100 novel_view.loss.w_schdl=bell \
    environment.slurm=False environment.resume=False logging.mode=none 



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hhor/len\${data.len}_w\${training.w_diffuse}_\${data.index}  \
    training=diffuse training.w_diffuse=0. \
    training.render_full_frame=False \
    data=hhor data.len=378,37,18 \
    environment.slurm=False environment.resume=False logging.mode=none 



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate_seq/hoi4d_len\${data.len}_w\${training.w_diffuse}_\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=1000 \
    data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259 \
    training.warmup=100,1000 novel_view.loss.w_schdl=bell \
    environment.slurm=False environment.resume=False logging.mode=none 




CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate_seq/hoi4d_len\${data.len}_w\${training.w_diffuse}_\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=1000 training.w_diffuse=0 \
    data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259 \
    environment.slurm=False environment.resume=False logging.mode=none 





---
# sure~~ confirmed
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=nothing_breaks/\${data.index}_w_\${training.w_diffuse}_clip\${training.clip}_geom\
    training=diffuse novel_view=geom novel_view.diff_name="hand_ddpm_geom/train_seg_CondGeomGlide" \
    training.w_diffuse=0.01 \
    data=ho3d  data.len=2 data.index=SM2_0001_dt02 \
    environment.slurm=False environment.resume=False logging.mode=none \


-


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate/sun_ablation_w\${training.w_diffuse}_\${novel_view.loss.w_schdl}_\${novel_view.sd_para.max_step}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d data.len=2 novel_view.sd_para.max_step=0.98 training.warmup=100 novel_view.loss.w_schdl=bell,dream \
    training.w_diffuse=1e-3 \
    environment.slurm=False environment.resume=False logging.mode=none 


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate/w\${training.w_diffuse}_\${novel_view.loss.w_schdl}_\${novel_view.sd_para.max_step}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=2 novel_view.sd_para.max_step=0.98,0.5 training.warmup=0,100 novel_view.loss.w_schdl=bell,dream \
    training.w_diffuse=1e-2,1e-3 \
    environment.slurm=False environment.resume=False logging.mode=none 
-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=uniform_rate/w\${training.w_diffuse}_\${novel_view.loss.w_schdl}_\${novel_view.sd_para.max_step}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="single_mode/cond_all_linear_0_0.5" \
    data=hoi4d data.len=2 novel_view.sd_para.max_step=0.98 training.warmup=0 novel_view.loss.w_schdl=bell \
    training.w_diffuse=0 \
    environment.slurm=False environment.resume=False logging.mode=none 
  



python -m tools.vis_diffusion --noise 0.9 --S 1 --load_pt

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=dev/tmp \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d \
    environment.slurm=True environment.slurm_timeout=720 
    
-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=ho3d_yana_det_revisit/\${data.index}_\${data.len}_w\${training.w_diffuse}_\${novel_view.sd_para.max_step}_warmup\${training.warmup} \
    training=diffuse novel_view=geom novel_view.diff_name="hand_ddpm_geom/train_seg_CondGeomGlide" \
    data=ho3d_det data.index=SM1_0360 training.w_diffuse=0 data.len=2,100 \
    environment.slurm=True \


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=ho3d_yana_det_revisit/\${data.index}_\${data.len}_w\${training.w_diffuse}_\${novel_view.sd_para.max_step}_warmup\${training.warmup} \
    training=diffuse novel_view=geom novel_view.diff_name="hand_ddpm_geom/train_seg_CondGeomGlide" \
    data=ho3d_det data.index=SM1_0360 training.w_diffuse=1e-2 data.len=2,100 \
    training.warmup=1000 novel_view.sd_para.max_step=0.5,0.1 \
    environment.slurm=True \

     environment.resume=False \


-

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d_warmup/w\${training.w_diffuse}_\${novel_view.sd_para.max_step}_warmup\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259,ZY20210800001_H1_C12_N31_S200_s02_T2_00052_00144 \
    training.warmup=1000 training.i_val=500 \
    training.w_diffuse=0   \
    environment.slurm=True environment.slurm_timeout=720 
-
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d_warmup/w\${training.w_diffuse}_\${novel_view.sd_para.max_step}_warmup\${training.warmup}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d data.index=ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159,ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259,ZY20210800001_H1_C12_N31_S200_s02_T2_00052_00144 \
    novel_view.sd_para.max_step=0.5,0.1 training.warmup=1000 training.i_val=500 \
    training.w_diffuse=1e-2   \
    environment.slurm=True enrironment.slurm_timeout=720 environment.resume=False logging.mode=none 



CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d_w_hoi4d_single_view/closer_look_\${novel_view.sd_param.max_step}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d data.len=2 novel_view.sd_param.max_step=0.98,0.5,0.1 \
    training.w_diffuse=1e-2 training.i_val=11 \
    environment.slurm=False environment.resume=False logging.mode=none 
  




CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=commit_mode/closer_C2_\${novel_view.sd_para.max_step}_\${training.warmup}_w\${training.w_diffuse}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4dC2_CondGeomGlide_1" \
    data=hoi4d data.len=2  \
    training.w_diffuse=0 training.i_val=11 training.warmup=0 \
    environment.slurm=True environment.resume=False logging.mode=none   
-
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=commit_mode/closer_C2_\${novel_view.sd_para.max_step}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4dC2_CondGeomGlide_1" \
    data=hoi4d data.len=2 novel_view.sd_para.max_step=0.98,0.5,0.1 \
    training.w_diffuse=1e-2 training.i_val=11 training.warmup=100,0 \
    environment.slurm=True environment.resume=False logging.mode=none   

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=commit_mode/closer_look_\${novel_view.sd_para.max_step}_\${training.warmup}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_sunday/hoi4d_CondGeomGlide_1" \
    data=hoi4d data.len=2 novel_view.sd_para.max_step=0.98 \
    training.w_diffuse=1e-3 training.warmup=100 \
    environment.slurm=True environment.resume=False logging.mode=none   


  data.index=ZY20210800002_H2_C5_N45_S261_s02_T2_00188_00259,ZY20210800001_H1_C12_N31_S200_s02_T2_00052_00144 \

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d_degrade_diffuse/cond_w\${training.w_diffuse}_\${data.len}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_only/hoi4d_CondGeomGlide" \
    training.ckpt_file=\${output}/hoi4d_w_hoi4d/cond_w0.01_1000_m0_n0_d0_ZY20210800001_H1_C2_N31_S92_s05_T2_00029_00159/ckpts/latest.pt \
    training.w_diffuse=1e-2,0,1e-3,1e-4 \
    data=hoi4d  \
    environment.slurm=True environment.resume=False  \


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m engine -m --config-name volsdf_nogt \
    expname=hoi4d_w_hoi4d/cond_w\${training.w_diffuse}_\${data.len}_m\${novel_view.loss.w_mask}_n\${novel_view.loss.w_normal}_d\${novel_view.loss.w_depth}_\${data.index}  \
    training=diffuse novel_view=geom novel_view.diff_name="ddpm_novel_only/hoi4d_CondGeomGlide" \
    novel_view.loss.w_mask=0,1 novel_view.loss.w_depth=0,1 novel_view.loss.w_normal=0,1 \
    data=hoi4d  \
    environment.slurm=True environment.resume=False



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