CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m tools.vis_clips  -m     load_folder=../org/w_normal/,../org/w_depth/,../org/w_mask/     \
    hydra/launcher=slurm



all fig for ours HOI4D

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m tools.vis_clips  -m     load_folder=../org/w_color/,../org/w_normal/,../org/w_depth/,../org/w_mask/,../org/ours/,../org/anneal/,../org/hand_prior/,../org/no_prior/,../org/obj_prior/     \
    hydra/launcher=slurm

all fig for ours wild

PYTHONPATH=. python -m tools.vis_clips  -m    \
 load_folder=../org/wild_ours   \
    hydra/launcher=slurm



# maek figure
# http://127.0.0.1:8520/figs_row/main/web/
python -m tools.make_better_fig --fig main --method gt,ours,ihoi,hhor --t 0 --degree 0,90 \


# make wild figure
python -m tools.make_better_fig --fig wild --method wild_ours,wild_ihoi --t 0 --degree overlay,90 --suf hoi,obj  --data wild

# maek ablate prior
python -m tools.make_better_fig --fig prior --method gt,ours,object_prior,hand_prior,no_prior --t 1 --degree 0,90 \
    --suf hoi

# maek ablate weight
python -m tools.make_better_fig --fig weight --method gt,ours,w_mask,w_normal \
    --suf hoi --t 2

# maek ablate prior
python -m tools.make_better_fig --fig anneal --method gt,ours,anneal --t 2 --degree 0,90 \
    --suf hoi


# make hhor figure
python -m tools.make_better_fig --fig hhor_more --method gt,hhor_ours,hhor_hhor  --degree 0,90 \
    --t 0 --data hhor --suf hoi


