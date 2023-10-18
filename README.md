# Diffusion-Guided Reconstruction of Everyday Hand-Object Interaction Clips
This is a barely cleaned version of DiffHOI. It is intended to quickly reproduce the result in paper. Better version will be released at [here](https://github.com/JudyYe/diffhoi).

[[Project]](https://judyye.github.io/diffhoi-www/)

## Quick start
- Installation 
```
# pytorch <= 1.10 to be compatible with FrankMocap
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

```
pip install -r requirements.txt
```

- Download pretrained diffusion model from [here]() and place it at `${environment.output}`

- Path specification: specify your output folder at `configs/environment/learn.yaml` (Even better practice is to create your own file `my_own.yaml` and append `environment=my_own` to the command in terminal)

```
${environment.output}/
  # pretrained diffusion model
  release/
    ddpm2d/
      checkpoints/
      config.yaml
  # Our test-time optimization results
  release_reproduce/
    Mug_1/
      ckpts/
      config.yaml
    Mug_2/
    ...

# preprocessed data
${envionment.output}/../
  HOI4D/
    Mug_1/
      cameras_hoi_smooth_100.npz
      image/
      mocap/
      ....
    Mug_2/
    ...
```


## Run on preprocessed HOI4D 
### Visualize Reconstruction
Suppose the models are unzipped to `${environment.output}/release_reproduce/`. The following command will render all models that matches `${load_folder}*` and save the rendering to `${load_folder}/SOME_MODEL/vis_clips`. 
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m tools.vis_clips  -m \
       load_folder=release_reproduce/     fig=True    
```
Note the slash in load_folder since the search pattern is `${load_folder}*`. 

Replace `fig=True` to `video=True` to render HOIs in videos.  More visualization options are at  `configs/eval.yaml` and `tools/vis_clips.py`. 

### Test-Time Optimization (~8 hours)
- For example, optimize a sequence under [`${environment.output}/../HOI4D/Kettle_1`](configs/data/hoi4d.yaml). The default values are specified in `configs/volsdf_nogt.yaml`.
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m train -m  \
    expname=dev/\${data.index}  \
    data.cat=Kettle data.ind=1  \
    environment=my_own   # see above about path specification
```

- Parameter sweep 
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m train -m  \
    expname=dev/\${data.index}  \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2  \
    environment=my_own   # see above about path specification
    hydra/launcher=slurm environment=grogu_judy # if you want to use slurm
```


## Run on custom data
### Preprocessing
1. Extract per-frame masks, hand boxes from videos by [this repo](https://github.com/JudyYe/hoi_vid).
2. Reconstruct hand poses; convert the extracted masks and poses to the data format ready to be consumed by DiffHOI.
    ```
    python -m  preprocess.inspect_custom --seq seq1,seq2,seq3 --out_dir save_path --inp_dir output_from_step1
    ``` 

  

### Test-Time Optimization (~8 hours)
The following command reconstruct preprocessed custom sequences under `${environment.output}/../${data.name}/${data_index}`. 

```
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m train -m  \
    expname=in_the_wild/\${data.name}\${data.index}  \
    data=custom data.name='1st_nocrop' \
    data.index=bottle_1,bottle_2 \
```


The command above assume your sequences are under structure:
```
${environment.output}/../
  1st_nocrop/
    bottle_1/
      image/
      ...
    bottle_2/
    ...
```