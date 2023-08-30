CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m tools.vis_clips  -m  \
    load_folder=../org/ours/  video=True   \

-
python -m tools.make_better_fig --fig vid_more --method ours  --data hoi4d \
    --suf vid_t 


python -m tools.make_better_fig --fig vid_homan --method gt,ours,homan_gt  --data hoi4d \
    --suf vid_t 



python -m tools.make_better_fig --fig wild --method gt,ours,homan --t 0 \
    --suf triplet  --data wild


CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m tools.vis_clips  -m  \
    load_folder=reproduce/  T_num=5 fig=True gt=True \


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m tools.vis_clips  -m  \
    load_folder=../org/ours/  video=True   \
    hydra/launcher=slurm


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m tools.vis_clips  -m  \
    load_folder=../org/wild_ours/  T_num=10   \
    hydra/launcher=slurm


# make wild figure
python -m tools.make_better_fig --fig wild --method wild_ours,wild_ihoi --t 0 \
    --suf triplet  --data wild

all fig for ours HOI4D

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m tools.vis_clips  -m    \
    load_folder=../org/w_normal/,../org/w_depth/,../org/w_mask/,../org/ours/,../org/hand_prior/,../org/no_prior/,../org/obj_prior/     \
    video=True \
    hydra/launcher=slurm


CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m tools.vis_clips  -m    \
    load_folder=../org/oTh/,../org/blend/ \
    video=True \
    hydra/launcher=slurm

-
all fig for ours wild

PYTHONPATH=. python -m tools.vis_clips  -m    \
 load_folder=../org/wild_ours/   \
 video=True \
    hydra/launcher=slurm



python -m tools.make_better_fig --fig vid_prior --method gt,ours,obj_prior,hand_prior,no_prior \
    --suf vid_t --data hoi4d_half &

python -m tools.make_better_fig --fig vid_main --method gt,ours,ihoi,hhor \
    --suf vid_t 

python -m tools.make_better_fig --fig vid_weight --method gt,ours,w_depth,w_mask,w_normal \
    --suf vid_t --data hoi4d_half

python -m tools.make_better_fig --fig vid_ablation --method gt,ours,blend,oTh \
    --suf vid_t --data hoi4d_half



python -m tools.make_better_fig --fig vid_wild --method wild_ours,wild_ihoi  --data wild \
    --suf vid_t 

python -m tools.make_better_fig --fig vid_more --method wild_ours  --data wild \
    --suf vid_t 



# maek figure
# http://127.0.0.1:8520/figs_row/main/web/
python -m tools.make_better_fig --fig main --method gt,ours,ihoi,hhor \
    --t 1 --suf triplet


# maek ablate prior
python -m tools.make_better_fig --fig prior --method gt,ours,obj_prior,hand_prior,no_prior \
    --t 2      --suf overlay_hoi,60_hoi


# maek ablate weight
python -m tools.make_better_fig --fig weight --method gt,ours,w_mask,w_normal \
    --t 0      --suf 0_hoi,60_hoi



# make hhor figure
python -m tools.make_better_fig --fig hhor_more --method gt,hhor_ours,hhor_hhor  --degree 0,90 \
    --t 0 --data hhor --suf hoi


python -m tools.make_better_fig --fig hhor_more --method gt,hhor_ours --t 0 --degree 270     --data hhor --suf hoi,obj
python -m tools.make_better_fig --fig hhor_more_hhor --method gt,hhor_hhor --t 0 --degree 90     --data hhor --suf hoi,obj
