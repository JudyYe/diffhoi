set -x 

cd ~/Tools/RAFT/


# 0.npz : 1--> 0
python demo.py --path /private/home/yufeiy2/vhoi/data/DTU/scan65/image/ --out /private/home/yufeiy2/vhoi/data/DTU/scan65/FlowBW --save_flow

# 0.npz: 0 --> 1
python demo.py --path /private/home/yufeiy2/vhoi/data/DTU/scan65/image/ --out /private/home/yufeiy2/vhoi/data/DTU/scan65/FlowFW --reverse --save_flow
