CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m ddpm.engine -m --config-name geom_glide  \
  expname=geom/ho3d_cam_\${ho3d.split} \
  model.model=GeomGlide ho3d.split=SM2,train_seg \
  environment.slurm=True \



CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m ddpm.engine -m --config-name geom_glide  \
  expname=geom/ho3d_cam_\${ho3d.split} \
  ho3d.split=SM2  model.model=GeomGlide \
  save_topk=-1 \
  environment.slurm=False \


CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m ddpm.engine -m --config-name sem_glide  \
  expname=ddpm2/glide_\${ho3d.split} \
  ho3d.split=SM2,train_seg \
  save_topk=-1 \
  environment.slurm=False \


CUDA_VISIBLE_DEVICES=1 python -m models.glide_base -m   \
  expname=dev/tmp \
  logging=none 


CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python engine.py -m --config-name sem_glide  \
  expname=ddpm/glide_\${ho3d.split} \
  ho3d.split=SM2,train_seg \
  save_topk=-1 \
  environment.slurm=True \
