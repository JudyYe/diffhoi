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