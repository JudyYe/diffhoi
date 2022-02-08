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