set -x 


CUDA_VISIBLE_DEVICES=1 python -m preprocess.inspect_hoi4d --render --skip & 
CUDA_VISIBLE_DEVICES=2 python -m preprocess.inspect_hoi4d --render --skip &
CUDA_VISIBLE_DEVICES=3 python -m preprocess.inspect_hoi4d --render --skip & 
CUDA_VISIBLE_DEVICES=4 python -m preprocess.inspect_hoi4d --render --skip &
CUDA_VISIBLE_DEVICES=5 python -m preprocess.inspect_hoi4d --render --skip & 
CUDA_VISIBLE_DEVICES=6 python -m preprocess.inspect_hoi4d --render --skip &
