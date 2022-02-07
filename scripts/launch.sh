

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