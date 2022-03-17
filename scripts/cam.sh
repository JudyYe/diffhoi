set -x 


data_list=('SMu41_0000' 'SS2_0000' 'BB12_0000' 'GSF11_0000' 'GSF11_1000' 'MC2_0000' 'SM2_0000')
data_list=('SMu41_0000' 'SS2_0000' 'BB12_0000' 'GSF11_0000' 'GSF11_1000' 'MC2_0000' 'SM2_0000')

for DATA in "${data_list[@]}"
do

python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/unknown_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --oTh:learn_R 1 --oTh:learn_t 1 --oTh:mode learn \
    --slurm --sl_ngpu 2 


python -m train --config configs/volsdf_hoi.yaml \
    --expname learn_oTh/gt_${DATA} --training:occ_mask indp  \
    --training:w_flow 0.0  --training:w_sdf 0.01 --data:scale_radius -1 --data:data_dir /checkpoint/yufeiy2/vhoi_out/syn_data/${DATA} \
    --slurm --sl_ngpu 2 

sleep 2

done