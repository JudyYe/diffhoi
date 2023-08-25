import os
import torch
import random
import numpy as np
from typing import Optional

import torch.distributed as dist

rank = 0        # process id, for IPC
local_rank = 0  # local GPU device id
world_size = 1  # number of processes

def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def init_env(args, gpu=None, ngpus_per_node=None):
    global rank, local_rank, world_size
    # while True:
    #     port = random.randint(10000, 20000)    
    #     if not is_port_in_use(port):
    #         print('use port', port)
    #         break
    port = args.environment.port
    print(gpu, ngpus_per_node)
    # if args.environment.multiprocessing_distributed:
        # return init_hydra(args, gpu, ngpus_per_node)

    if args.ddp:
        #------------- multi process running, using DDP
        if 'SLURM_PROCID' in os.environ:
            #--------- for SLURM
            slurm_initialize('nccl', port=port)
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            #--------- for torch.distributed.launch
            rank = args.environment.rank * ngpus_per_node + gpu
            local_rank = rank
            world_size = args.environment.world_size

            dist.init_process_group(backend='nccl', 
            init_method="tcp://{}:{}".format("localhost", str(port)),
            rank=local_rank,        world_size = world_size)
        dist.barrier()
        torch.cuda.set_device(local_rank)
        args.device_ids = [local_rank]
        print("=> Init Env @ DDP: rank={}, world_size={}, local_rank={}.\n\tdevice_ids set to {}".format(rank, world_size, local_rank, args.device_ids))
        # NOTE: important!
    else:
        #------------- single process running, using single GPU or DataParallel
        # torch.cuda.set_device(args.device_ids[0])
        print("=> Init Env @ single process: use device_ids = {}".format(args.device_ids))
        rank = 0
        local_rank = args.device_ids[0]
        world_size = 1
        torch.cuda.set_device(args.device_ids[0])
    set_seed(args.seed)


def slurm_initialize(backend='nccl', port: Optional[int] = None):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "13333"
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl',
        init_method="tcp://{}:{}".format("localhost", port),
        world_size=ntasks,
        rank=proc_id,
)
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    os.environ['LOCAL_RANK'] = str(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_master():
    return rank == 0

def get_rank():
    return int(os.environ.get('SLURM_PROCID', rank))

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', local_rank))

def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', world_size))